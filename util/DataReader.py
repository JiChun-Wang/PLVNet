import os, json, math

import numpy as np
from PIL import Image
import cv2
from transforms3d.quaternions import quat2mat


class DataReader(object) :
	def __init__(self, params_file_name) :
		'''
		Input:
			params_file_name: path of parameter file ("parameter.json")
			target_num_point: target number of sampled points, default is 16384
		'''
		self.params = self._load_parameters(params_file_name)
		location, rotation = self.params['location'], self.params['rotation']
		cam_in_world = np.identity(4)
		cam_in_world[:3, :3] = quat2mat(rotation)
		cam_in_world[:3, 3] = location
		self.cam_in_world = cam_in_world

		cam_paras = self.params
		self.cam_K = np.array([[cam_paras['fu'], 0., cam_paras['cu']],
		                       [0., cam_paras['fv'], cam_paras['cv']],
		                       [0., 0., 1.]])
		self.H, self.W = cam_paras['resolutionY'], cam_paras['resolutionX']
		self.resolution = cam_paras['resolution_big']

	def _load_parameters(self, params_file_name):
		'''
		Input:
			params_file_name: path of parameter file ("parameter.json")
		'''
		with open(params_file_name, 'r') as f :
			config = json.load(f)
			params = config
			# compute fu and fv
			angle = params['perspectiveAngle'] * math.pi / 180.0
			f = 1.0 / (2 * math.tan(angle / 2.0)) * params['resolution_big']
			params['fu'] = f
			params['fv'] = f
			params['cu'] = params['resolution_big'] / 2.0
			params['cv'] = params['cu']
			params['max_val_in_depth'] = 65535.0
		return params

	def read_depth_map(self, depth_path):
		depth_img = cv2.imread(depth_path, -1)
		assert depth_img.shape == (self.params['resolutionY'], self.params['resolutionX']) and depth_img.dtype == np.uint16
		camera_info = self.params
		clip_start = camera_info['clip_start']
		clip_end = camera_info['clip_end']
		depth = (clip_start + (depth_img / float(camera_info['max_val_in_depth'])) * (clip_end - clip_start))
		depth[depth < 0.1] = 0
		depth[depth > 3.] = 0
		return depth

	def read_normal_image(self, normal_path) :
		normal = np.array(Image.open(normal_path))
		normal = normal / 255.0 * 2 - 1
		valid_mask = np.linalg.norm(normal, axis=-1) > 0.1
		normal = normal / (np.linalg.norm(normal, axis=-1)[:, :, None] + 1e-15)
		normal[valid_mask == 0] = 0
		return normal.astype(np.float32)

	def read_segment_map(self, segment_path):
		segment = cv2.imread(segment_path, -1).astype(int)
		return segment

	def depth2xyzmap(self, depth):
		K = self.cam_K
		invalid_mask = (depth < 0.1)
		H, W = depth.shape[:2]
		vs, us = np.meshgrid(np.arange(0, H), np.arange(0, W), sparse=False, indexing='ij')
		vs = vs.reshape(-1) + self.params['pixelOffset_Y_KoSyTopLeft']
		us = us.reshape(-1) + self.params['pixelOffset_X_KoSyTopLeft']
		zs = depth.reshape(-1)
		xs = - (us - K[0, 2]) * zs / K[0, 0]
		ys = - (vs - K[1, 2]) * zs / K[1, 1]
		pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N,3)
		xyz_map = pts.reshape(H, W, 3).astype(np.float32)
		xyz_map[invalid_mask] = 0
		return xyz_map.astype(np.float32)

	def read_gt_file(self, gt_path):
		poses = {}
		visibility_rate = {}
		with open(gt_path, 'r') as f:
			for line in f.readlines()[1 :]:
				line = line.strip()
				if len(line) == 0:
					continue
				words = line.split(',')
				id = int(words[0])
				if id > 0:
					location = list(map(float, words[2 :5]))
					rotation = np.array(list(map(float, words[5:14]))).reshape((3, 3)).T
					pose = np.identity(4)
					pose[:3, :3] = rotation
					pose[:3, 3] = location
					poses[id] = pose
					visibility_rate[id] = float(words[-1])

		meta= {
			'cam_in_world': self.cam_in_world,
			'poses': poses,
			'visibility_rate': visibility_rate
		}
		return meta