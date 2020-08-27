import torch
import torch.optim as optim

import time
import argparse
import math

from dataset import CamLocDataset
from network import Network

parser = argparse.ArgumentParser(
	description='Initialize a scene coordinate regression network.',
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('scene', help='name of a scene in the dataset folder')

parser.add_argument('network', help='output file name for the network')

parser.add_argument('--learningrate', '-lr', type=float, default=0.0001, 
	help='learning rate')

parser.add_argument('--iterations', '-iter', type=int, default=1000000,
	help='number of training iterations, i.e. numer of model updates')

parser.add_argument('--inittolerance', '-itol', type=float, default=0.1, 
	help='switch to reprojection error optimization when predicted scene coordinate is within this tolerance threshold to the ground truth scene coordinate, in meters')

parser.add_argument('--mindepth', '-mind', type=float, default=0.1, 
	help='enforce  predicted scene coordinates to be this far in front of the camera plane, in meters')

parser.add_argument('--maxdepth', '-maxd', type=float, default=1000, 
	help='enforce that scene coordinates are at most this far in front of the camera plane, in meters')

parser.add_argument('--targetdepth', '-td', type=float, default=10, 
	help='if ground truth scene coordinates are unknown, use a proxy scene coordinate on the pixel ray with this distance from the camera, in meters')

parser.add_argument('--softclamp', '-sc', type=float, default=100, 
	help='robust square root loss after this threshold, in pixels')

parser.add_argument('--hardclamp', '-hc', type=float, default=1000, 
	help='clamp loss with this threshold, in pixels')

parser.add_argument('--mode', '-m', type=int, default=1, choices=range(3),
	help='training mode: 0 = RGB only (no ground truth scene coordinates), 1 = RGB + ground truth scene coordinates, 2 = RGB-D')

parser.add_argument('--sparse', '-sparse', action='store_true',
	help='for mode 1 (RGB + ground truth scene coordinates) use sparse scene coordinate initialization targets (eg. for Cambridge) instead of rendered depth maps (eg. for 7scenes and 12scenes).')

parser.add_argument('--session', '-sid', default='',
	help='custom session name appended to output files, useful to separate different runs of a script')

opt = parser.parse_args()

use_init = opt.mode > 0

# for RGB-D initialization, we utilize ground truth scene coordinates as in mode 2 (RGB + ground truth scene coordinates)
trainset = CamLocDataset("./datasets/" + opt.scene + "/train", mode=min(opt.mode, 1), sparse=opt.sparse, augment=True)
trainset_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=6)

print("Found %d training images for %s." % (len(trainset), opt.scene))

print("Calculating mean scene coordinate for the scene...")

mean = torch.zeros((3))
count = 0

for image, gt_pose, gt_coords, focal_length, file in trainset_loader:

	if use_init:
		# use mean of ground truth scene coordinates

		gt_coords = gt_coords[0]
		gt_coords = gt_coords.view(3, -1)

		coord_mask = gt_coords.abs().sum(0) > 0
		if coord_mask.sum() > 0:
			gt_coords = gt_coords[:, coord_mask]

			mean += gt_coords.sum(1)
			count += coord_mask.sum()
	else:
		# use mean of camera position
		mean += gt_pose[0, 0:3, 3]
		count += 1
	
mean /= count

print("Done. Mean: %.2f, %.2f, %.2f\n" % (mean[0], mean[1], mean[2]))

# create network
network = Network(mean)
network = network.cuda()
network.train()

optimizer = optim.Adam(network.parameters(), lr=opt.learningrate)

iteration = 0
epochs = int(opt.iterations / len(trainset))

# keep track of training progress
train_log = open('log_init_%s_%s.txt' % (opt.scene, opt.session), 'w', 1)

# generate grid of target reprojection pixel positions
pixel_grid = torch.zeros((2, 
	math.ceil(5000 / network.OUTPUT_SUBSAMPLE),		# 5000px is max limit of image size, increase if needed
	math.ceil(5000 / network.OUTPUT_SUBSAMPLE)))

for x in range(0, pixel_grid.size(2)):
	for y in range(0, pixel_grid.size(1)):
		pixel_grid[0, y, x] = x * network.OUTPUT_SUBSAMPLE + network.OUTPUT_SUBSAMPLE / 2
		pixel_grid[1, y, x] = y * network.OUTPUT_SUBSAMPLE + network.OUTPUT_SUBSAMPLE / 2

pixel_grid = pixel_grid.cuda()

for epoch in range(epochs):	

	print("=== Epoch: %d ======================================" % epoch)

	for image, gt_pose, gt_coords, focal_length, file in trainset_loader:

		start_time = time.time()

		# create camera calibartion matrix
		focal_length = float(focal_length[0])
		cam_mat = torch.eye(3)
		cam_mat[0, 0] = focal_length
		cam_mat[1, 1] = focal_length
		cam_mat[0, 2] = image.size(3) / 2
		cam_mat[1, 2] = image.size(2) / 2
		cam_mat = cam_mat.cuda()

		scene_coords = network(image.cuda()) 

		# calculate loss dependant on the mode

		if opt.mode == 2:
			# === RGB-D mode, optimize 3D distance to ground truth scene coordinates ======

			scene_coords = scene_coords[0].view(3, -1)
			gt_coords = gt_coords[0].view(3, -1).cuda()

			# check for invalid ground truth scene coordinates
			gt_coords_mask = gt_coords.abs().sum(0) > 0

			loss = torch.norm(scene_coords - gt_coords, dim=0, p=2)[gt_coords_mask]
			loss = loss.mean()
			num_valid_sc = gt_coords_mask.float().mean()

		else:
			# === RGB mode, optmize a variant of the reprojection error ===================

			# crop ground truth pixel positions to prediction size
			pixel_grid_crop = pixel_grid[:,0:scene_coords.size(2),0:scene_coords.size(3)].clone()
			pixel_grid_crop = pixel_grid_crop.view(2, -1)

			# make 3D points homogeneous
			ones = torch.ones((scene_coords.size(0), 1, scene_coords.size(2), scene_coords.size(3)))
			ones = ones.cuda()

			scene_coords = torch.cat((scene_coords, ones), 1)
			scene_coords = scene_coords.squeeze().view(4, -1)

			# prepare pose for projection operation
			gt_pose = gt_pose[0].inverse()[0:3,:]
			gt_pose = gt_pose.cuda()

			# scene coordinates to camera coordinate 
			camera_coords = torch.mm(gt_pose, scene_coords)

			# re-project predicted scene coordinates
			reprojection_error = torch.mm(cam_mat, camera_coords)
			reprojection_error[2].clamp_(min=opt.mindepth) # avoid division by zero
			reprojection_error = reprojection_error[0:2] / reprojection_error[2]

			reprojection_error = reprojection_error - pixel_grid_crop
			reprojection_error = reprojection_error.norm(2, 0)

			# check predicted scene coordinate for various constraints
			invalid_min_depth = camera_coords[2] < opt.mindepth # behind or too close to camera plane
			invalid_repro = reprojection_error > opt.hardclamp # check for very large reprojection errors

			if use_init:
				# ground truth scene coordinates available, transform to uniform
				gt_coords = torch.cat((gt_coords.cuda(), ones), 1)		
				gt_coords = gt_coords.squeeze().view(4, -1)

				# check for invalid/unknown ground truth scene coordinates (all zeros)
				gt_coords_mask = torch.abs(gt_coords[0:3]).sum(0) == 0

				# scene coordinates to camera coordinate 
				target_camera_coords = torch.mm(gt_pose, gt_coords)

				# distance between predicted and ground truth coordinates
				gt_coord_dist = torch.norm(camera_coords - target_camera_coords, dim=0, p=2)

				# check for additional constraints regarding ground truth scene coordinates
				invalid_gt_distance = gt_coord_dist > opt.inittolerance # too far from ground truth scene coordinates
				invalid_gt_distance[gt_coords_mask] = 0  # filter unknown ground truth scene coordinates

				# combine all constraints
				valid_scene_coordinates = (invalid_min_depth + invalid_gt_distance + invalid_repro) == 0

			else:
				# no ground truth scene coordinates available, enforce max distance of predicted coordinates
				invalid_max_depth = camera_coords[2] > opt.maxdepth

				# combine all constraints
				valid_scene_coordinates = (invalid_min_depth + invalid_max_depth + invalid_repro) == 0

			num_valid_sc = int(valid_scene_coordinates.sum())

			# assemble loss
			loss = 0
				
			if num_valid_sc > 0:

				# reprojection error for all valid scene coordinates
				reprojection_error = reprojection_error[valid_scene_coordinates]

				# calculate soft clamped l1 loss of reprojection error
				loss_l1= reprojection_error[reprojection_error <= opt.softclamp]
				loss_sqrt = reprojection_error[reprojection_error > opt.softclamp]
				loss_sqrt = torch.sqrt(opt.softclamp*loss_sqrt)

				loss += (loss_l1.sum() + loss_sqrt.sum())

			if num_valid_sc < scene_coords.size(1):

				invalid_scene_coordinates = (valid_scene_coordinates == 0) 

				if use_init:
					# 3D distance loss for all invalid scene coordinates where the ground truth is known
					invalid_scene_coordinates[gt_coords_mask] = 0 

					loss += gt_coord_dist[invalid_scene_coordinates].sum() 
				else:
					# generate proxy coordinate targets with constant depth assumption
					target_camera_coords = pixel_grid_crop
					target_camera_coords[0] -= image.size(3) / 2
					target_camera_coords[1] -= image.size(2) / 2
					target_camera_coords *= opt.targetdepth
					target_camera_coords /= focal_length
					# make homogeneous
					target_camera_coords = torch.cat((target_camera_coords, torch.ones((1, target_camera_coords.size(1))).cuda()), 0)

					# distance 
					loss += torch.abs(camera_coords[:,invalid_scene_coordinates] - target_camera_coords[:, invalid_scene_coordinates]).sum()

			loss /= scene_coords.size(1)
			num_valid_sc /= scene_coords.size(1)

		loss.backward()			# calculate gradients (pytorch autograd)
		optimizer.step()		# update all model parameters
		optimizer.zero_grad()

		print('Iteration: %6d, Loss: %.1f, Valid: %.1f%%, Time: %.2fs' % (iteration, loss, num_valid_sc*100, time.time()-start_time), flush=True)
		train_log.write('%d %f %f\n' % (iteration, loss, num_valid_sc))

		iteration = iteration + 1

		del loss

	print('Saving snapshot of the network to %s.' % opt.network)
	torch.save(network.state_dict(), opt.network)
	

print('Done without errors.')
train_log.close()
