import os, glob, pickle, time, gzip
import random
import json
import math

import numpy as np
import cv2
import open3d as o3d
import torch
from torch.utils.data import Dataset
from transforms3d.quaternions import quat2mat

from ..util.utils import toOpen3dCloud, defor_2D, defor_2D_full
from .data_augmentation import *


class PoseDataset(Dataset):
	def __init__(self, cfg, phase='train'):
		super().__init__()
		self.cfg = cfg
		assert phase in ['train', 'val'], 'The dataset is only for training or validating! '
		self.phase = phase

		data_dir = cfg.data_root
		dataset = cfg.dataset
		obj_name = cfg.obj_name
		self.num_point = cfg.num_pts

		ply_path = f'{data_dir}/{dataset}/{obj_name}/mesh.ply'
		pcd = o3d.io.read_point_cloud(ply_path)
		self.model_size = np.asarray(pcd.points).max(0)

		if phase == 'train':
			self.files = sorted(glob.glob(f"{data_dir}/{dataset}/{obj_name}/train_pose_estimation/*.pkl"))[
						 :cfg.num_train]
		else:
			self.files = sorted(glob.glob(f"{data_dir}/{dataset}/{obj_name}/val_pose_estimation/*.pkl"))[:cfg.num_val]

		print("phase: {}, num files={}".format(phase, len(self.files)))

	def __len__(self):
		return len(self.files)

	def data_augmentation_center(self, xyz, drop_ratio=0.5, max_drop_ratio=0.5, depth_noise=0.005):
		## randomly dropout
		if np.random.uniform() < drop_ratio:
			num_pts = len(xyz)
			dropout_ratio = np.random.uniform(0, max_drop_ratio)
			n_drop = int(dropout_ratio * num_pts)
			drop_ids = np.random.choice(num_pts, size=n_drop, replace=False)
			keep_ids = list(set(np.arange(num_pts)) - set(drop_ids))
			xyz = xyz[keep_ids]

		## gaussian noise
		std = np.random.uniform(0, depth_noise)
		xyz += np.random.normal(0, std, size=xyz.shape)
		return xyz

	def data_augmentation_edge(self, xyz, min_drop_ratio=0, max_drop_ratio=0.8, depth_noise=0.005):
		num_pts = len(xyz)

		## dropout
		dropout_ratio = np.random.uniform(min_drop_ratio, max_drop_ratio)
		n_drop = int(dropout_ratio * num_pts)
		drop_ids = np.random.choice(num_pts, size=n_drop, replace=False)
		keep_ids = list(set(np.arange(num_pts)) - set(drop_ids))
		xyz = xyz[keep_ids]

		## gaussian noise
		std = np.random.uniform(0, depth_noise)
		xyz += np.random.normal(0, std, size=xyz.shape)
		return xyz

	def transform(self, data):
		cloud_xyz = data['cloud_xyz']
		cloud_seg = data['cloud_seg']
		rot = data['rotation']
		tran = data['translation']

		if self.phase == 'train':
			cloud_center = cloud_xyz[cloud_seg == 1]
			cloud_edge = cloud_xyz[cloud_seg == 2]
			cloud_center = self.data_augmentation_center(cloud_center)
			cloud_edge = self.data_augmentation_edge(cloud_edge)
			cloud_xyz = np.concatenate([cloud_center, cloud_edge], axis=0)

		replace = len(cloud_xyz) < self.num_point
		ids = np.random.choice(np.arange(len(cloud_xyz)), size=self.cfg.num_pts, replace=replace)
		cloud_xyz = cloud_xyz[ids].reshape(-1, 3)

		# if self.phase == 'train':
		# 	# cloud_xyz = GaussianNoise()(cloud_xyz)
		# 	# cloud_xyz = PointCloudShuffle(self.num_point)(cloud_xyz)

		rt_aug_t, rt_aug_R = self.generate_aug_parameters()

		data_dict = {}
		data_dict['points'] = torch.as_tensor(cloud_xyz.astype(np.float32)).contiguous()
		data_dict['Rs'] = torch.as_tensor(rot.astype(np.float32)).contiguous()
		data_dict['ts'] = torch.as_tensor(tran.astype(np.float32)).contiguous()
		data_dict['aug_rt_t'] = torch.as_tensor(rt_aug_t, dtype=torch.float32).contiguous()
		data_dict['aug_rt_R'] = torch.as_tensor(rt_aug_R, dtype=torch.float32).contiguous()
		data_dict['size'] = torch.as_tensor(self.model_size, dtype=torch.float32).contiguous()
		return data_dict

	def generate_aug_parameters(self, ax=5, ay=5, az=5, a=15):
		# for R, t aug
		Rm = get_rotation(np.random.uniform(-a, a), np.random.uniform(-a, a), np.random.uniform(-a, a))
		dx = np.random.rand() * 2 * ax - ax
		dy = np.random.rand() * 2 * ay - ay
		dz = np.random.rand() * 2 * az - az
		return np.array([dx, dy, dz], dtype=np.float32) / 1000.0, Rm

	def __getitem__(self, index):
		file = self.files[index]
		while 1:
			try:
				with gzip.open(file, 'rb') as ff:
					data = pickle.load(ff)
				break
			except Exception as e:
				time.sleep(0.001)

		data = self.transform(data)
		return data


class CatPoseDataset(Dataset):
	def __init__(self, cfg, phase='train'):
		super().__init__()
		self.cfg = cfg
		assert phase in ['train', 'val'], 'The dataset is only for training or validating! '
		self.phase = phase

		data_dir = cfg.data_root
		dataset = cfg.dataset
		obj_name = cfg.obj_name
		self.num_point = cfg.num_pts
		if phase == 'train':
			self.files = sorted(glob.glob(f"{data_dir}/{dataset}/{obj_name}/train_pose_estimation/*.pkl"))[
						 :cfg.num_train]
		else:
			self.files = sorted(glob.glob(f"{data_dir}/{dataset}/{obj_name}/val_pose_estimation/*.pkl"))[:cfg.num_val]

		print("phase: {}, num files={}".format(phase, len(self.files)))

	def __len__(self):
		return len(self.files)

	def transform(self, data):
		cloud_xyz = data['cloud_xyz']
		rot = data['rotation']
		tran = data['translation']
		fsnet_scale = data['fsnet_scale']
		mean_shape = data['mean_shape']

		replace = len(cloud_xyz) < self.num_point
		ids = np.random.choice(np.arange(len(cloud_xyz)), size=self.cfg.num_pts, replace=replace)
		cloud_xyz = cloud_xyz[ids].reshape(-1, 3)

		rt_aug_t, rt_aug_R = self.generate_aug_parameters()

		data_dict = {}
		data_dict['point'] = torch.as_tensor(cloud_xyz.astype(np.float32)).contiguous()
		data_dict['rotation'] = torch.as_tensor(rot.astype(np.float32)).contiguous()
		data_dict['translation'] = torch.as_tensor(tran.astype(np.float32)).contiguous()
		data_dict['fsnet_scale'] = torch.as_tensor(fsnet_scale.astype(np.float32)).contiguous()
		data_dict['mean_shape'] = torch.as_tensor(mean_shape.astype(np.float32)).contiguous()
		data_dict['aug_rt_t'] = torch.as_tensor(rt_aug_t, dtype=torch.float32).contiguous()
		data_dict['aug_rt_R'] = torch.as_tensor(rt_aug_R, dtype=torch.float32).contiguous()

		return data_dict

	def generate_aug_parameters(self, ax=10, ay=10, az=10, a=15):
		# for R, t aug
		Rm = get_rotation(np.random.uniform(-a, a), np.random.uniform(-a, a), np.random.uniform(-a, a))
		dx = np.random.rand() * 2 * ax - ax
		dy = np.random.rand() * 2 * ay - ay
		dz = np.random.rand() * 2 * az - az
		return np.array([dx, dy, dz], dtype=np.float32) / 1000.0, Rm

	def __getitem__(self, index):
		file = self.files[index]
		while 1:
			try:
				with gzip.open(file, 'rb') as ff:
					data = pickle.load(ff)
				break
			except Exception as e:
				time.sleep(0.001)

		data = self.transform(data)
		return data


class PoseDataset_v2(Dataset):
	def __init__(self, cfg, phase='train'):
		super().__init__()
		assert phase in ['train'], 'The dataset is only for training'
		self.phase = phase
		self.cfg = cfg

		data_dir = cfg.data_root
		dataset = cfg.dataset
		obj_name = cfg.obj_name

		self.params = self._load_parameters(os.path.join(data_dir, dataset, obj_name, 'parameter.json'))
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

		if phase == 'train':
			self.files = sorted(glob.glob(f"{data_dir}/{dataset}/{obj_name}_train/train_part_*/gt/cycle_*/*.csv"))
		else:
			raise NotImplemented

		print("phase: {}, num files={}".format(phase, len(self.files)))

	def __len__(self):
		return len(self.files)

	def _load_parameters(self, params_file_name):
		'''
		Input:
			params_file_name: path of parameter file ("parameter.json")
		'''
		with open(params_file_name, 'r') as f:
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
		assert depth_img.shape == (self.H, self.W) and depth_img.dtype == np.uint16
		camera_info = self.params
		clip_start = camera_info['clip_start']
		clip_end = camera_info['clip_end']
		depth = (clip_start + (depth_img / 65535.0) * (clip_end - clip_start))
		depth[depth < 0.1] = 0
		depth[depth > 3.] = 0
		return depth

	def read_segment_map(self, segment_path):
		segment = cv2.imread(segment_path, -1).astype(np.uint8)
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
			for line in f.readlines()[1:]:
				line = line.strip()
				if len(line) == 0:
					continue
				words = line.split(',')
				id = int(words[0])
				if id > 0:
					location = list(map(float, words[2:5]))
					rotation = np.array(list(map(float, words[5:14]))).reshape((3, 3)).T
					pose = np.identity(4)
					pose[:3, :3] = rotation
					pose[:3, 3] = location
					poses[id] = pose
					visibility_rate[id] = float(words[-1])

		meta = {
			'cam_in_world': self.cam_in_world,
			'poses': poses,
			'visibility_rate': visibility_rate
		}
		return meta

	def __getitem__(self, index):
		meta_path = self.files[index]
		if not os.path.exists(meta_path):
			return self.__getitem__((index + 1) % self.__len__())
		dpt_path = meta_path.replace('/gt/', '/p_depth/').replace('.csv', '_depth_uint16.png')
		if not os.path.exists(dpt_path):
			return self.__getitem__((index + 1) % self.__len__())
		seg_path = meta_path.replace('/gt/', '/p_segmentation/').replace('.csv', '_segmentation.png')
		if not os.path.exists(seg_path):
			return self.__getitem__((index + 1) % self.__len__())

		dpt_map = self.read_depth_map(dpt_path)
		seg_map = self.read_segment_map(seg_path)
		xyz_map = self.depth2xyzmap(dpt_map)
		meta = self.read_gt_file(meta_path)

		choose_ids = []
		seg_ids = np.unique(seg_map)
		for seg_id in seg_ids:
			if seg_id in [0]:
				continue
			if meta['visibility_rate'][seg_id] > 0.5:
				choose_ids.append(seg_id)

		if len(choose_ids) == 0:
			return self.__getitem__((index + 1) % self.__len__())

		choose_id = random.choice(choose_ids)
		roi_mask = (seg_map == choose_id).astype('uint8')

		roi_mask = defor_2D(roi_mask)
		roi_mask_def = defor_2D_full(roi_mask)

		mask = (roi_mask_def != 0) & (dpt_map >= 0.1)
		cloud_xyz = xyz_map[mask].reshape(-1, 3)

		replace = len(cloud_xyz) < self.cfg.num_pts
		ids = np.random.choice(np.arange(len(cloud_xyz)), size=self.cfg.num_pts, replace=replace)
		cloud_xyz = cloud_xyz[ids].reshape(-1, 3)
		# pcd = toOpen3dCloud(cloud_xyz)
		# dwn_pcd = pcd.farthest_point_down_sample(num_samples=self.cfg.num_pts)
		# cloud_xyz = np.asarray(dwn_pcd.points).reshape(-1, 3)

		if self.phase == 'train':
			cloud_xyz = GaussianNoise()(cloud_xyz)

		ob_in_cam = meta['poses'][choose_id]
		rotation, translation = ob_in_cam[:3, :3], ob_in_cam[:3, 3]

		rt_aug_t, rt_aug_R = self.generate_aug_parameters()

		data_dict = {}
		data_dict['points'] = torch.as_tensor(cloud_xyz.astype(np.float32)).contiguous()
		data_dict['Rs'] = torch.as_tensor(rotation.astype(np.float32)).contiguous()
		data_dict['ts'] = torch.as_tensor(translation.astype(np.float32)).contiguous()
		data_dict['aug_rt_t'] = torch.as_tensor(rt_aug_t, dtype=torch.float32).contiguous()
		data_dict['aug_rt_R'] = torch.as_tensor(rt_aug_R, dtype=torch.float32).contiguous()
		return data_dict

	def generate_aug_parameters(self, ax=5, ay=5, az=5, a=15):
		# for R, t aug
		Rm = get_rotation(np.random.uniform(-a, a), np.random.uniform(-a, a), np.random.uniform(-a, a))
		dx = np.random.rand() * 2 * ax - ax
		dy = np.random.rand() * 2 * ay - ay
		dz = np.random.rand() * 2 * az - az
		return np.array([dx, dy, dz], dtype=np.float32) / 1000.0, Rm


if __name__ == '__main__':
	def visualization_line(points_original, R, T):
		print(len(points_original))
		ids = np.random.choice(np.arange(len(points_original)), size=50, replace=False)
		points = points_original[ids].reshape(-1, 3)

		points_line_cano = np.zeros([len(points), 3], dtype=np.float32)
		points_cano = (R.T @ (points.T - t.reshape(-1, 1))).T
		points_line_cano[:, 2] = points_cano[:, 2]
		points_plane = (R @ points_line_cano.T + t.reshape(-1, 1)).T

		vote_points = np.concatenate([points, points_plane], axis=0)
		vote_lines = [[i, i + len(points)] for i in range(len(points))]
		vote_color = [[1, 0, 0] for i in range(len(vote_lines))]

		gt_green_v = R[:, 2] * 0.1 + T
		pose_points = np.concatenate([T.reshape(1, -1), gt_green_v.reshape(1, -1)], axis=0)
		pose_lines = [[0, 1]]
		pose_color = [[0, 1, 0] for i in range(len(pose_lines))]

		total_points = np.concatenate([vote_points, pose_points], axis=0)
		total_lines = vote_lines + [[2 * len(points), 2 * len(points) + 1]]
		total_color = vote_color + pose_color

		points_pcd = o3d.geometry.PointCloud()
		points_pcd.points = o3d.utility.Vector3dVector(total_points)
		points_pcd.paint_uniform_color([0, 0.3, 0])

		# 绘制线条
		lines_pcd = o3d.geometry.LineSet()
		lines_pcd.lines = o3d.utility.Vector2iVector(total_lines)
		lines_pcd.colors = o3d.utility.Vector3dVector(total_color)  # 线条颜色
		lines_pcd.points = o3d.utility.Vector3dVector(total_points)

		pcd = toOpen3dCloud(points_original)
		o3d.visualization.draw_geometries([pcd, points_pcd, lines_pcd])


	def visualization_plane(points_original, R, T, on_plane='xy'):
		print(len(points_original))
		ids = np.random.choice(np.arange(len(points_original)), size=200, replace=False)
		points = points_original[ids].reshape(-1, 3)

		if on_plane == 'xy':
			points_plane_cano = np.zeros([len(points), 3], dtype=np.float32)
			points_cano = (R.T @ (points.T - t.reshape(-1, 1))).T
			points_plane_cano[:, :2] = points_cano[:, :2]
			points_plane = (R @ points_plane_cano.T + t.reshape(-1, 1)).T

			vote_points = np.concatenate([points, points_plane], axis=0)
			vote_lines = [[i, i + len(points)] for i in range(len(points))]
			vote_color = [[1, 0, 0] for i in range(len(vote_lines))]
		elif on_plane == 'yz':
			points_plane_cano = np.zeros([len(points), 3], dtype=np.float32)
			points_cano = (R.T @ (points.T - t.reshape(-1, 1))).T
			points_plane_cano[:, 1:] = points_cano[:, 1:]
			points_plane = (R @ points_plane_cano.T + t.reshape(-1, 1)).T

			vote_points = np.concatenate([points, points_plane], axis=0)
			vote_lines = [[i, i + len(points)] for i in range(len(points))]
			vote_color = [[1, 0, 0] for i in range(len(vote_lines))]

		gt_green_v = R[:, 2] * 0.1 + T
		gt_red_v = R[:, 0] * 0.1 + T
		pose_points = np.concatenate([T.reshape(1, -1), gt_green_v.reshape(1, -1), gt_red_v.reshape(1, -1)], axis=0)
		pose_lines = [[0, 1], [0, 2]]
		pose_color = [[0, 0, 1] for i in range(len(pose_lines))]

		total_points = np.concatenate([vote_points, pose_points], axis=0)
		total_lines = vote_lines + [[2 * len(points), 2 * len(points) + 1], [2 * len(points), 2 * len(points) + 2]]
		total_color = vote_color + pose_color

		points_pcd = o3d.geometry.PointCloud()
		points_pcd.points = o3d.utility.Vector3dVector(total_points)
		points_pcd.paint_uniform_color([0, 0.3, 0])

		# 绘制线条
		lines_pcd = o3d.geometry.LineSet()
		lines_pcd.lines = o3d.utility.Vector2iVector(total_lines)
		lines_pcd.colors = o3d.utility.Vector3dVector(total_color)  # 线条颜色
		lines_pcd.points = o3d.utility.Vector3dVector(total_points)

		pcd = toOpen3dCloud(points_original)
		o3d.visualization.draw_geometries([pcd, points_pcd, lines_pcd])


	import open3d as o3d
	from ..util.config import get_parser

	cfg = get_parser()
	dataset = PoseDataset(cfg, 'val')
	idx = random.randrange(0, len(dataset))
	data = dataset[idx]
	points_original = data['points'].numpy()
	R = data['Rs'].numpy()
	t = data['ts'].numpy()
	print(len(points_original))
	print(points_original[2048])
	print(t)

	# from transformations import euler_matrix
	#
	# Rs = []
	# for rz in [0, np.pi] :
	# 	tf = euler_matrix(0, 0, rz, axes='sxyz')
	# 	Rs.append(tf[:3, :3])
	#
	# print(Rs)
	visualization_line(points_original, R, t)

# pcd = toOpen3dCloud(points_original)
# o3d.visualization.draw_geometries([pcd])
