import numpy as np
from PIL import Image
import cv2
from scipy.spatial import cKDTree
import open3d as o3d


def toOpen3dCloud(points, colors=None, normals=None):
	cloud = o3d.geometry.PointCloud()
	cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
	if colors is not None:
		if colors.max() > 1:
			colors = colors / 255.0
		cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
	if normals is not None:
		cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
	return cloud


def depth2xyzmap(depth, K):
	invalid_mask = (depth < 0.1)
	H, W = depth.shape[:2]
	vs, us = np.meshgrid(np.arange(0, H), np.arange(0, W), sparse=False, indexing='ij')
	vs = vs.reshape(-1)
	us = us.reshape(-1)
	zs = depth.reshape(-1)
	xs = (us - K[0, 2]) * zs / K[0, 0]
	ys = (vs - K[1, 2]) * zs / K[1, 1]
	pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N,3)
	xyz_map = pts.reshape(H, W, 3).astype(np.float32)
	xyz_map[invalid_mask] = 0
	return xyz_map.astype(np.float32)


def depth2xyzmap_w_offset(depth, K, offset_x, offset_y):
	invalid_mask = (depth < 0.1)
	H, W = depth.shape[:2]
	vs, us = np.meshgrid(np.arange(0, H), np.arange(0, W), sparse=False, indexing='ij')
	vs = vs.reshape(-1) + offset_y
	us = us.reshape(-1) + offset_x
	zs = depth.reshape(-1)
	xs = - (us - K[0, 2]) * zs / K[0, 0]
	ys = - (vs - K[1, 2]) * zs / K[1, 1]
	pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N,3)
	xyz_map = pts.reshape(H, W, 3).astype(np.float32)
	xyz_map[invalid_mask] = 0
	return xyz_map.astype(np.float32)


def read_normal_image(img_dir):
	normal = np.array(Image.open(img_dir))
	normal = normal / 255.0 * 2 - 1
	valid_mask = np.linalg.norm(normal, axis=-1) > 0.1
	normal = normal / (np.linalg.norm(normal, axis=-1)[:, :, None] + 1e-15)
	normal[valid_mask == 0] = 0
	return normal.astype(np.float32)


def correct_pcd_normal_direction(pcd, view_port=np.zeros((3), dtype=float)):
	view_dir = view_port.reshape(-1, 3) - np.asarray(pcd.points)  # (N,3)
	view_dir = view_dir / np.linalg.norm(view_dir, axis=1).reshape(-1, 1)
	normals = np.asarray(pcd.normals) / (np.linalg.norm(np.asarray(pcd.normals), axis=1) + 1e-10).reshape(-1, 1)
	dots = (view_dir * normals).sum(axis=1)
	indices = np.where(dots < 0)
	normals[indices, :] = -normals[indices, :]
	pcd.normals = o3d.utility.Vector3dVector(normals)
	return pcd


def parse_camera_config(file) :
	with open(file) as f :
		lines = f.readlines()

	cam_params = {}
	for i in range(len(lines)) :
		line = lines[i].split()
		if line[0] == 'location' :
			value = [float(line[1]), float(line[2]), float(line[3])]
		elif line[0] == 'rotation' :
			value = [float(line[1]), float(line[2]), float(line[3]), float(line[4])]
		else :
			if line[0] in ['width', 'height'] :
				value = int(line[1])
			else :
				value = float(line[1])
		cam_params[line[0]] = value
	return cam_params


def read_depth_map(path, clip_start, clip_end) :
	depth = cv2.imread(path, -1) / 65535.0
	depth = clip_start + (clip_end - clip_start) * depth
	return depth
# def read_depth_map(path, clip_start, clip_end) :
# 	with Image.open(path) as di:
# 		depth = np.asarray(di) /
# 	depth = clip_start + (clip_end - clip_start) * depth
# 	return depth


def fill_depth_normal(cloud_xyz):
	pcd = toOpen3dCloud(cloud_xyz)
	pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.003, max_nn=30))
	pcd = correct_pcd_normal_direction(pcd)
	normals = np.asarray(pcd.normals).copy()
	return normals.astype(np.float32)


def cloud_sample(cloud_xyz, downsample_size=0.005):
	pcd = toOpen3dCloud(cloud_xyz)
	downpcd = pcd.voxel_down_sample(voxel_size=downsample_size)
	pts = np.asarray(downpcd.points).copy()
	cloud_xyz, indices = find_nearest_points(cloud_xyz, pts)
	return cloud_xyz, indices


def find_nearest_points(src_xyz, query):
	kdtree = cKDTree(src_xyz)
	dists, indices = kdtree.query(query)
	query_xyz = src_xyz[indices]
	return query_xyz, indices


def defor_2D(roi_mask, min_drop_ratio=0, max_drop_ratio=0.8):
	'''

	:param roi_mask: 256 x 256
	:param rand_r: randomly expand or shrink the mask iter rand_r
	:return:
	'''
	roi_mask = roi_mask.copy().squeeze()
	# if np.random.rand() > rand_pro:
	# 	return roi_mask
	mask = roi_mask.copy()
	kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
	mask_erode = cv2.erode(mask, kernel_erode, iterations=4)  # rand_r
	change_list = roi_mask[mask_erode != mask]
	l_list = change_list.size
	if l_list < 1.0:
		return roi_mask
	dropout_ratio = np.random.uniform(min_drop_ratio, max_drop_ratio)
	n_drop = int(dropout_ratio * l_list)
	choose = np.random.choice(l_list, n_drop, replace=False)
	change_list = np.ones_like(change_list)
	change_list[choose] = 0.0
	roi_mask[mask_erode != mask] = change_list
	roi_mask[roi_mask > 0.0] = 1.0
	return roi_mask


class GaussianNoise(object):
	def __init__(self, depth_noise=0.002, prob=0.8):
		self.depth_noise = depth_noise
		self.prob = prob

	def __call__(self, depth):
		mask = depth>100
		if np.random.uniform() < self.prob:
			std = np.random.uniform(0, self.depth_noise)
			noise = np.random.normal(0, std, size=depth.shape)
			depth[mask] = depth[mask] + noise[mask]
		return depth


def to_homo(pts):
	'''
	@pts: (N,3 or 2) will homogeneliaze the last dimension
	'''
	assert len(pts.shape)==2, f'pts.shape: {pts.shape}'
	homo = np.concatenate((pts, np.ones((pts.shape[0],1))),axis=-1)
	return homo


def kabsch(P, Q):
	"""
	Using the Kabsch algorithm with two sets of paired point P and Q, centered
	around the centroid. Each vector set is represented as an NxD
	matrix, where D is the the dimension of the space.
	The algorithm works in three steps:
	- a centroid translation of P and Q (assumed done before this function
	  call)
	- the computation of a covariance matrix C
	- computation of the optimal rotation matrix U
	For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
	Parameters
	----------
	P : array
		(N,D) matrix, where N is points and D is dimension.
	Q : array
		(N,D) matrix, where N is points and D is dimension.
	Returns
	-------
	U : matrix
		Rotation matrix (D,D)
	"""

	# Computation of the covariance matrix
	#print(P.shape,Q.shape)
	# print(np.mean(P,0))
	# P= P-np.mean(P,0)
	# Q =Q - np.mean(Q, 0)
	# print(P)
	# tests
	C = np.dot(P.T, Q)

	# Computation of the optimal rotation matrix
	# This can be done using singular value decomposition (SVD)
	# Getting the sign of the det(V)*(W) to decide
	# whether we need to correct our rotation matrix to ensure a
	# right-handed coordinate system.
	# And finally calculating the optimal rotation matrix U
	# see http://en.wikipedia.org/wiki/Kabsch_algorithm
	U, S, V = np.linalg.svd(C)
	#S=np.diag(S)
	#print(C)
	# print(S)
	#print(np.dot(U,np.dot(S,V)))
	d = (np.linalg.det(V.T) * np.linalg.det(U.T)) <0.0

	# d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

	# E = np.diag(np.array([1, 1, 1]))
	# if d:
	#     S[-1] = -S[-1]
	#     V[:, -1] = -V[:, -1]
	E = np.diag(np.array([1, 1, (np.linalg.det(V.T) * np.linalg.det(U.T))]))


	# print(E)

	# Create Rotation matrix U
	#print(V)
	#print(U)
	R = np.dot(V.T ,np.dot(E,U.T))

	return R