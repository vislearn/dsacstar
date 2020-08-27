import torch
import torch.optim as optim

import argparse
import time
import random

import dsacstar

from dataset import CamLocDataset
from network import Network

parser = argparse.ArgumentParser(
	description='Train scene coordinate regression in an end-to-end fashion.',
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('scene', help='name of a scene in the dataset folder')

parser.add_argument('network_in', help='file name of a network initialized for the scene')

parser.add_argument('network_out', help='output file name for the new network')

parser.add_argument('--hypotheses', '-hyps', type=int, default=64, 
	help='number of hypotheses, i.e. number of RANSAC iterations')

parser.add_argument('--threshold', '-t', type=float, default=10, 
	help='inlier threshold in pixels')

parser.add_argument('--inlieralpha', '-ia', type=float, default=100, 
	help='alpha parameter of the soft inlier count; controls the softness of the hypotheses score distribution; lower means softer')

parser.add_argument('--learningrate', '-lr', type=float, default=0.000001, 
	help='learning rate')

parser.add_argument('--iterations', '-it', type=int, default=100000, 
	help='number of training iterations, i.e. network parameter updates')

parser.add_argument('--weightrot', '-wr', type=float, default=1.0, 
	help='weight of rotation part of pose loss')

parser.add_argument('--weighttrans', '-wt', type=float, default=100.0, 
	help='weight of translation part of pose loss')

parser.add_argument('--softclamp', '-sc', type=float, default=100, 
	help='robust square root loss after this threshold')

parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100, 
	help='maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when checking pose consistency towards all measurements; error is clamped to this value for stability')

parser.add_argument('--mode', '-m', type=int, default=1, choices=[1,2],
	help='test mode: 1 = RGB, 2 = RGB-D')

parser.add_argument('--session', '-sid', default='',
	help='custom session name appended to output files. Useful to separate different runs of the program')

opt = parser.parse_args()

trainset = CamLocDataset("./datasets/" + opt.scene + "/train", mode=(0 if opt.mode < 2 else opt.mode), augment=True, aug_rotation=0, aug_scale_min=1, aug_scale_max=1) # use only photometric augmentation, not rotation and scaling
trainset_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=6)

print("Found %d training images for %s." % (len(trainset), opt.scene))

# load network
network = Network(torch.zeros((3)))
network.load_state_dict(torch.load(opt.network_in))
network = network.cuda()
network.train()

print("Successfully loaded %s." % opt.network_in)

optimizer = optim.Adam(network.parameters(), lr=opt.learningrate)

iteration = 0
epochs = int(opt.iterations / len(trainset))

# keep track of training progress
train_log = open('log_e2e_%s_%s.txt' % (opt.scene, opt.session), 'w', 1)

for epoch in range(epochs):	

	print("=== Epoch: %7d ======================================" % epoch)

	for image, pose, camera_coordinates, focal_length, file in trainset_loader:

		start_time = time.time()

		focal_length = float(focal_length[0])
		pose = pose[0]

		# predict scene coordinates
		scene_coordinates = network(image.cuda())
		scene_coordinate_gradients = torch.zeros(scene_coordinates.size())

		if opt.mode == 2:
			# RGB-D mode
			loss = dsacstar.backward_rgbd(
				scene_coordinates.cpu(), 
				camera_coordinates,
				scene_coordinate_gradients,
				pose, 
				opt.hypotheses, 
				opt.threshold,
				opt.weightrot,
				opt.weighttrans,
				opt.softclamp,
				opt.inlieralpha,
				opt.maxpixelerror,
				random.randint(0,1000000)) # used to initialize random number generator in C++

		else:
			# RGB mode
			loss = dsacstar.backward_rgb(
				scene_coordinates.cpu(), 
				scene_coordinate_gradients,
				pose, 
				opt.hypotheses, 
				opt.threshold,
				focal_length, 
				float(image.size(3) / 2), #principal point assumed in image center
				float(image.size(2) / 2),
				opt.weightrot,
				opt.weighttrans,
				opt.softclamp,
				opt.inlieralpha,
				opt.maxpixelerror,
				network.OUTPUT_SUBSAMPLE,
				random.randint(0,1000000)) # used to initialize random number generator in C++

		# update network parameters
		torch.autograd.backward((scene_coordinates), (scene_coordinate_gradients.cuda()))
		optimizer.step()
		optimizer.zero_grad()
		
		end_time = time.time()-start_time
		print('Iteration: %6d, Loss: %.2f, Time: %.2fs \n' % (iteration, loss, end_time), flush=True)

		train_log.write('%d %f\n' % (iteration, loss))
		iteration = iteration + 1

	print('Saving snapshot of the network to %s.' % opt.network_out)
	torch.save(network.state_dict(), opt.network_out)

print('Done without errors.')
train_log.close()
