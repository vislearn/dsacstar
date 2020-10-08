import os
import numpy as np
import random
import math

from skimage import io
from skimage import color
from skimage.transform import rotate, resize

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from network import Network

class CamLocDataset(Dataset):
	"""Camera localization dataset.

	Access to image, calibration and ground truth data given a dataset directory.
	"""

	def __init__(self, root_dir, 
		mode=1, 
		sparse=False, 
		augment=False, 
		aug_rotation=30, 
		aug_scale_min=2/3, 
		aug_scale_max=3/2, 
		aug_contrast=0.1, 
		aug_brightness=0.1, 
		image_height=480):
		'''Constructor.

		Parameters:
			root_dir: Folder of the data (training or test).
			mode: 
				0 = RGB only, load no initialization targets, 
				1 = RGB + ground truth scene coordinates, load or generate ground truth scene coordinate targets
				2 = RGB-D, load camera coordinates instead of scene coordinates
			sparse: for mode = 1 (RGB+GT SC), load sparse initialization targets when True, load dense depth maps and generate initialization targets when False
			augment: Use random data augmentation, note: not supported for mode = 2 (RGB-D) since pre-generateed eye coordinates cannot be agumented
			aug_rotation: Max 2D image rotation angle, sampled uniformly around 0, both directions
			aug_scale_min: Lower limit of image scale factor for uniform sampling
			aug_scale_min: Upper limit of image scale factor for uniform sampling
			aug_contrast: Max relative scale factor for image contrast sampling, e.g. 0.1 -> [0.9,1.1]
			aug_brightness: Max relative scale factor for image brightness sampling, e.g. 0.1 -> [0.9,1.1]
			image_height: RGB images are rescaled to this maximum height
		'''

		self.init = (mode == 1)
		self.sparse = sparse
		self.eye = (mode == 2)

		self.image_height = image_height

		self.augment = augment
		self.aug_rotation = aug_rotation
		self.aug_scale_min = aug_scale_min
		self.aug_scale_max = aug_scale_max
		self.aug_contrast = aug_contrast
		self.aug_brightness = aug_brightness
		
		if self.eye and self.augment and (self.aug_rotation > 0 or self.aug_scale_min != 1 or self.aug_scale_max != 1):
			print("WARNING: Check your augmentation settings. Camera coordinates will not be augmented.")


		rgb_dir = root_dir + '/rgb/'
		pose_dir =  root_dir + '/poses/'
		calibration_dir = root_dir + '/calibration/'
		if self.eye:
			coord_dir =  root_dir + '/eye/'
		elif self.sparse:
			coord_dir =  root_dir + '/init/'
		else:
			coord_dir =  root_dir + '/depth/'

		self.rgb_files = os.listdir(rgb_dir)
		self.rgb_files = [rgb_dir + f for f in self.rgb_files]
		self.rgb_files.sort()

		self.image_transform = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize(self.image_height),
			transforms.Grayscale(),
			transforms.ToTensor(),
			transforms.Normalize(
				mean=[0.4], # statistics calculated over 7scenes training set, should generalize fairly well
				std=[0.25]
				)
			])

		self.pose_files = os.listdir(pose_dir)
		self.pose_files = [pose_dir + f for f in self.pose_files]
		self.pose_files.sort()

		self.pose_transform = transforms.Compose([
			transforms.ToTensor()
			])

		self.calibration_files = os.listdir(calibration_dir)
		self.calibration_files = [calibration_dir + f for f in self.calibration_files]
		self.calibration_files.sort()		

		if self.init or self.eye:
			self.coord_files = os.listdir(coord_dir)
			self.coord_files = [coord_dir + f for f in self.coord_files]
			self.coord_files.sort()

		if len(self.rgb_files) != len(self.pose_files):
			raise Exception('RGB file count does not match pose file count!')

		if not sparse:

			#create grid of 2D pixel positions when generating scene coordinates from depth
			self.prediction_grid = np.zeros((2, 
				math.ceil(5000 / Network.OUTPUT_SUBSAMPLE), 
				math.ceil(5000 / Network.OUTPUT_SUBSAMPLE)))

			for x in range(0, self.prediction_grid.shape[2]):
				for y in range(0, self.prediction_grid.shape[1]):
					self.prediction_grid[0, y, x] = x * Network.OUTPUT_SUBSAMPLE
					self.prediction_grid[1, y, x] = y * Network.OUTPUT_SUBSAMPLE		

	def __len__(self):
		return len(self.rgb_files)

	def __getitem__(self, idx):

		image = io.imread(self.rgb_files[idx])

		if len(image.shape) < 3:
			image = color.gray2rgb(image)

		focal_length = float(np.loadtxt(self.calibration_files[idx]))

		pose = np.loadtxt(self.pose_files[idx])
		pose = torch.from_numpy(pose).float()

		if self.init:
			if self.sparse:
				coords = torch.load(self.coord_files[idx])
			else:
				depth = io.imread(self.coord_files[idx])
				depth = depth.astype(np.float64)
				depth /= 1000 # from millimeters to meters
		elif self.eye: 
			coords = torch.load(self.coord_files[idx])
		else:
			coords = 0

		if self.augment:

			scale_factor = random.uniform(self.aug_scale_min, self.aug_scale_max)
			angle = random.uniform(-self.aug_rotation, self.aug_rotation)

			# augment input image
			cur_image_transform = transforms.Compose([
				transforms.ToPILImage(),
				transforms.Resize(int(self.image_height * scale_factor)),
				transforms.Grayscale(),
				transforms.ColorJitter(brightness=self.aug_brightness, contrast=self.aug_contrast),
				transforms.ToTensor(),
				transforms.Normalize(
					mean=[0.4],
					std=[0.25]
					)
			])
			image = cur_image_transform(image)	

			# scale focal length
			focal_length *= scale_factor

			# rotate input image
			def my_rot(t, angle, order, mode='constant'):
				t = t.permute(1,2,0).numpy()
				t = rotate(t, angle, order=order, mode=mode)
				t = torch.from_numpy(t).permute(2, 0, 1).float()
				return t

			image = my_rot(image, angle, 1, 'reflect')

			if self.init:

				if self.sparse:
					#rotate and scale initalization targets
					coords_w = math.ceil(image.size(2) / Network.OUTPUT_SUBSAMPLE)
					coords_h = math.ceil(image.size(1) / Network.OUTPUT_SUBSAMPLE)
					coords = F.interpolate(coords.unsqueeze(0), size=(coords_h, coords_w))[0]

					coords = my_rot(coords, angle, 0)
				else:
					#rotate and scale depth maps
					depth = resize(depth, image.shape[1:], order=0)
					depth = rotate(depth, angle, order=0, mode='constant')

			# rotate ground truth camera pose
			angle = angle * math.pi / 180
			pose_rot = torch.eye(4)
			pose_rot[0, 0] = math.cos(angle)
			pose_rot[0, 1] = -math.sin(angle)
			pose_rot[1, 0] = math.sin(angle)
			pose_rot[1, 1] = math.cos(angle)

			pose = torch.matmul(pose, pose_rot)			

		else:

			image = self.image_transform(image)	

		if self.init and not self.sparse:
			#generate initialization targets from depth map
			
			offsetX = int(Network.OUTPUT_SUBSAMPLE/2)
			offsetY = int(Network.OUTPUT_SUBSAMPLE/2)

			coords = torch.zeros((
				3, 
				math.ceil(image.shape[1] / Network.OUTPUT_SUBSAMPLE), 
				math.ceil(image.shape[2] / Network.OUTPUT_SUBSAMPLE)))

			# subsample to network output size
			depth = depth[offsetY::Network.OUTPUT_SUBSAMPLE,offsetX::Network.OUTPUT_SUBSAMPLE] 

			# construct x and y coordinates of camera coordinate
			xy = self.prediction_grid[:,:depth.shape[0],:depth.shape[1]].copy()
			# add random pixel shift
			xy[0] += offsetX
			xy[1] += offsetY
			# substract principal point (assume image center)
			xy[0] -= image.shape[2] / 2
			xy[1] -= image.shape[1] / 2
			# reproject
			xy /= focal_length
			xy[0] *= depth
			xy[1] *= depth

			#assemble camera coordinates trensor
			eye = np.ndarray((4, depth.shape[0], depth.shape[1]))
			eye[0:2] = xy
			eye[2] = depth
			eye[3] = 1

			# eye to scene coordinates
			sc = np.matmul(pose.numpy(), eye.reshape(4,-1))
			sc = sc.reshape(4, depth.shape[0], depth.shape[1])

			# mind pixels with invalid depth
			sc[:, depth == 0] = 0
			sc[:, depth > 1000] = 0
			sc = torch.from_numpy(sc[0:3])

			coords[:,:sc.shape[1],:sc.shape[2]] = sc

		return image, pose, coords, focal_length, self.rgb_files[idx]
