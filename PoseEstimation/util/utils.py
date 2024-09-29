import numpy as np
from transformations import euler_matrix
import open3d as o3d
import cv2
import torch


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


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


def defor_2D(roi_mask, min_drop_ratio=0, max_drop_ratio=0.5):
	'''

	:param roi_mask: 256 x 256
	:param rand_r: randomly expand or shrink the mask iter rand_r
	:return:
	'''
	roi_mask = roi_mask.copy().squeeze()
	# if np.random.rand() > rand_pro:
	# 	return roi_mask
	mask = roi_mask.copy()
	kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	mask_erode = cv2.erode(mask, kernel_erode, iterations=3)  # rand_r
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


def defor_2D_full(roi_mask, min_drop_ratio=0, max_drop_ratio=0.5):
	'''

	:param roi_mask: 256 x 256
	:param rand_r: randomly expand or shrink the mask iter rand_r
	:return:
	'''
	roi_mask = roi_mask.copy().squeeze()
	if np.random.rand() > 0.5:
		return roi_mask
	mask = roi_mask.copy()
	change_list = roi_mask[mask!=0]
	l_list = change_list.size
	if l_list < 1.0:
		return roi_mask
	dropout_ratio = np.random.uniform(min_drop_ratio, max_drop_ratio)
	n_drop = int(dropout_ratio * l_list)
	choose = np.random.choice(l_list, n_drop, replace=False)
	change_list = np.ones_like(change_list)
	change_list[choose] = 0.0
	roi_mask[mask!=0] = change_list
	roi_mask[roi_mask > 0.0] = 1.0
	return roi_mask


def get_symmetry_tfs(class_name):
	tfs = []
	if class_name == 'brick':
		for rz in [0, np.pi]:
			tf = euler_matrix(0, 0, rz, axes='sxyz')
			tfs.append(tf[:3, :3])
	elif class_name == 'tless_20':
		for rz in [0, np.pi]:
			tf = euler_matrix(0, 0, rz, axes='sxyz')
			tfs.append(tf[:3, :3])
	elif class_name == 'tless_29':
		for rz in [0, np.pi]:
			tf = euler_matrix(0, 0, rz, axes='sxyz')
			tfs.append(tf[:3, :3])
	else:
		tf = np.identity(4)
		tfs.append(tf[:3, :3])

	return np.array(tfs)


def get_gt_v(Rs, axis=2):
	bs = Rs.shape[0]  # bs x 3 x 3
	# TODO use 3 axis, the order remains: do we need to change order?
	if axis == 3:
		corners = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=torch.float).to(Rs.device)
		corners = corners.view(1, 3, 3).repeat(bs, 1, 1)  # bs x 3 x 3
		gt_vec = torch.bmm(Rs, corners).transpose(2, 1).reshape(bs, -1)
	else:
		assert axis == 2
		corners = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=torch.float).to(Rs.device)
		corners = corners.view(1, 3, 3).repeat(bs, 1, 1)  # bs x 3 x 3
		gt_vec = torch.bmm(Rs, corners).transpose(2, 1).reshape(bs, -1)
	gt_green = gt_vec[:, 3:6]
	gt_red = gt_vec[:, (6, 7, 8)]
	return gt_green, gt_red