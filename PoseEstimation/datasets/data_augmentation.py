import math
import numpy as np


class NormalizeCloud:
	def __init__(self):
		pass

	def __call__(self, cloud_xyz):
		max_xyz = cloud_xyz.max(axis=0)
		min_xyz = cloud_xyz.min(axis=0)
		scale = (max_xyz - min_xyz).max()
		cloud_xyz = (cloud_xyz - min_xyz) / (scale + 1e-15)
		return cloud_xyz


class CentralizedCloud:
	def __init__(self):
		pass

	def __call__(self, cloud_xyz):
		return cloud_xyz - cloud_xyz.mean(axis=0)


class GaussianNoise:
	def __init__(self, depth_noise=0.005):
		self.depth_noise = depth_noise

	def __call__(self, cloud_xyz):
		std = np.random.uniform(0, self.depth_noise)
		cloud_xyz += np.random.normal(0, self.depth_noise, size=cloud_xyz.shape)
		return cloud_xyz


class DropoutCloud:
	def __init__(self, drop_ratio=0.5, drop_max_ratio=0.25):
		self.drop_ratio = drop_ratio
		self.drop_max_ratio = drop_max_ratio

	def __call__(self, data):
		if np.random.uniform() < self.drop_ratio:
			dropout_ratio = np.random.uniform(0, self.drop_max_ratio)
			n_drop = int(dropout_ratio * len(data))
			drop_ids = np.random.choice(len(data), size=n_drop, replace=False)
			keep_ids = list(set(np.arange(len(data))) - set(drop_ids))
			to_replace_ids = np.random.choice(keep_ids, size=n_drop, replace=True)
			data[drop_ids] = data[to_replace_ids]
			# cloud_nml[drop_ids] = cloud_nml[to_replace_ids]
		return data


class PointCloudShuffle:
	def __init__(self, num_point):
		self.num_point = num_point

	def __call__(self, xyz, rgb):
		pt_idx = np.arange(0, self.num_point)
		np.random.shuffle(pt_idx)
		return xyz[pt_idx], rgb[pt_idx]


def get_rotation(x_, y_, z_):
	# print(math.cos(math.pi/2))
	x = float(x_ / 180) * math.pi
	y = float(y_ / 180) * math.pi
	z = float(z_ / 180) * math.pi
	R_x = np.array([[1, 0, 0],
					[0, math.cos(x), -math.sin(x)],
					[0, math.sin(x), math.cos(x)]])

	R_y = np.array([[math.cos(y), 0, math.sin(y)],
					[0, 1, 0],
					[-math.sin(y), 0, math.cos(y)]])

	R_z = np.array([[math.cos(z), -math.sin(z), 0],
					[math.sin(z), math.cos(z), 0],
					[0, 0, 1]])
	return np.dot(R_z, np.dot(R_y, R_x)).astype(np.float32)


# class DropoutCloud:
# 	def __init__(self, drop_max_ratio=0.5):
# 		self.drop_max_ratio = drop_max_ratio
#
# 	def __call__(self, cloud_xyz, cloud_nml):
# 		dropout_ratio = np.random.uniform(0, self.drop_max_ratio)
# 		n_drop = int(dropout_ratio * len(cloud_xyz))
# 		drop_ids = np.random.choice(len(cloud_xyz), size=n_drop, replace=False)
# 		keep_ids = list(set(np.arange(len(cloud_xyz))) - set(drop_ids))
# 		cloud_xyz = cloud_xyz[keep_ids]
# 		cloud_nml = cloud_nml[keep_ids]
# 		return cloud_xyz, cloud_nml
