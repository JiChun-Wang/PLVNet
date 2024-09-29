import os, time
code_dir = os.path.dirname(os.path.realpath(__file__))
import math
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d

# from PoseEstimation.util.config import cfg
from PoseEstimation.network.GSCPose import GSCPose_Re, GSCPose
from PoseEstimation.datasets.dataset_pose import PoseDataset
from PoseEstimation.util.utils import toOpen3dCloud
from util.align_line import estimate3DLine, linear_fitting_3D_points


def refine_zx_vec(z, x):
	##  z, x are rotation vectors
	z = z.view(-1)
	x = x.view(-1)
	rot_y = torch.cross(z, x)
	rot_y = rot_y / (torch.norm(rot_y) + 1e-8)
	# cal angle between z and x
	z_x_cos = torch.sum(z * x)
	z_x_theta = torch.acos(z_x_cos)
	theta = 0.5 * (z_x_theta - math.pi / 2)
	# first rotate z
	c = torch.cos(theta)
	s = torch.sin(theta)
	rotmat_z = torch.tensor([[rot_y[0]*rot_y[0]*(1-c)+c, rot_y[0]*rot_y[1]*(1-c)-rot_y[2]*s, rot_y[0]*rot_y[2]*(1-c)+rot_y[1]*s],
							 [rot_y[1]*rot_y[0]*(1-c)+rot_y[2]*s, rot_y[1]*rot_y[1]*(1-c)+c, rot_y[1]*rot_y[2]*(1-c)-rot_y[0]*s],
							 [rot_y[0]*rot_y[2]*(1-c)-rot_y[1]*s, rot_y[2]*rot_y[1]*(1-c)+rot_y[0]*s, rot_y[2]*rot_y[2]*(1-c)+c]]).to(z.device)
	new_z = torch.mm(rotmat_z, z.view(-1, 1))
	# then rotate x
	c = torch.cos(-theta)
	s = torch.sin(-theta)
	rotmat_x = torch.tensor([[rot_y[0]*rot_y[0]*(1-c)+c, rot_y[0]*rot_y[1]*(1-c)-rot_y[2]*s, rot_y[0]*rot_y[2]*(1-c)+rot_y[1]*s],
							 [rot_y[1]*rot_y[0]*(1-c)+rot_y[2]*s, rot_y[1]*rot_y[1]*(1-c)+c, rot_y[1]*rot_y[2]*(1-c)-rot_y[0]*s],
							 [rot_y[0]*rot_y[2]*(1-c)-rot_y[1]*s, rot_y[2]*rot_y[1]*(1-c)+rot_y[0]*s, rot_y[2]*rot_y[2]*(1-c)+c]]).to(x.device)

	new_x = torch.mm(rotmat_x, x.view(-1, 1))
	return new_z.view(-1), new_x.view(-1)


def refine_zy_vec(z, y):
	##  z, y are rotation vectors
	z = z.view(-1)
	y = y.view(-1)
	rot_x = torch.cross(y, z)
	rot_x = rot_x / (torch.norm(rot_x) + 1e-8)
	# cal angle between z and y
	z_y_cos = torch.sum(z * y)
	z_y_theta = torch.acos(z_y_cos)
	theta = 0.5 * (z_y_theta - math.pi / 2)
	# first rotate z
	c = torch.cos(-theta)
	s = torch.sin(-theta)
	rotmat_z = torch.tensor([[rot_x[0]*rot_x[0]*(1-c)+c, rot_x[0]*rot_x[1]*(1-c)-rot_x[2]*s, rot_x[0]*rot_x[2]*(1-c)+rot_x[1]*s],
							 [rot_x[1]*rot_x[0]*(1-c)+rot_x[2]*s, rot_x[1]*rot_x[1]*(1-c)+c, rot_x[1]*rot_x[2]*(1-c)-rot_x[0]*s],
							 [rot_x[0]*rot_x[2]*(1-c)-rot_x[1]*s, rot_x[2]*rot_x[1]*(1-c)+rot_x[0]*s, rot_x[2]*rot_x[2]*(1-c)+c]]).to(z.device)
	new_z = torch.mm(rotmat_z, z.view(-1, 1))
	# then rotate y
	c = torch.cos(theta)
	s = torch.sin(theta)
	rotmat_y = torch.tensor([[rot_x[0]*rot_x[0]*(1-c)+c, rot_x[0]*rot_x[1]*(1-c)-rot_x[2]*s, rot_x[0]*rot_x[2]*(1-c)+rot_x[1]*s],
							 [rot_x[1]*rot_x[0]*(1-c)+rot_x[2]*s, rot_x[1]*rot_x[1]*(1-c)+c, rot_x[1]*rot_x[2]*(1-c)-rot_x[0]*s],
							 [rot_x[0]*rot_x[2]*(1-c)-rot_x[1]*s, rot_x[2]*rot_x[1]*(1-c)+rot_x[0]*s, rot_x[2]*rot_x[2]*(1-c)+c]]).to(y.device)

	new_y = torch.mm(rotmat_y, y.view(-1, 1))
	return new_z.view(-1), new_y.view(-1)


def get_x_vec(z):
	z = z.view(-1)
	x = torch.tensor([z[1], -z[0], 0]).to(z.device)
	x = x / (torch.norm(x) + 1e-8)
	return z.view(-1), x.view(-1)


def get_y_vec(z):
	z = z.view(-1)
	y = torch.tensor([z[1], -z[0], 0]).to(z.device)
	y = y / (torch.norm(y) + 1e-8)
	return z.view(-1), y.view(-1)


def get_rot_mat_zx(z, x):
	# poses

	z = F.normalize(z, p=2, dim=-1)  # bx3
	y = torch.cross(z, x, dim=-1)  # bx3
	y = F.normalize(y, p=2, dim=-1)  # bx3
	x = torch.cross(y, z, dim=-1)  # bx3

	# (*,3)x3 --> (*,3,3)
	return torch.stack((x, y, z), dim=-1)  # (b,3,3)


def get_rot_mat_zy(z, y):
	# poses

	z = F.normalize(z, p=2, dim=-1)  # bx3
	x = torch.cross(y, z, dim=-1)  # bx3
	x = F.normalize(x, p=2, dim=-1)  # bx3
	y = torch.cross(z, x, dim=-1)  # bx3

	# (*,3)x3 --> (*,3,3)
	return torch.stack((x, y, z), dim=-1)  # (b,3,3)


def visualization_line(points_original, p_green_R, p_red_R, p_T, p_Vote) :
	print(len(points_original))
	ids = np.random.choice(np.arange(len(points_original)), size=50, replace=False)
	points = points_original[ids].reshape(-1, 3)

	points_line = points + p_Vote[ids]
	vote_points = np.concatenate([points, points_line], axis=0)
	vote_lines = [[i, i + len(points)] for i in range(len(points))]
	vote_color = [[1, 0, 0] for i in range(len(vote_lines))]

	gt_green_v = p_green_R * 0.1 + p_T
	gt_red_v = p_red_R * 0.1 + p_T
	pose_points = np.concatenate([p_T.reshape(1, -1), gt_green_v.reshape(1, -1), gt_red_v.reshape(1, -1)], axis=0)
	pose_lines = [[0, 1], [0, 2]]
	pose_color = [[0, 1, 0] for i in range(len(pose_lines))]

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


def visualization_plane(points_original, p_green_R, p_red_R, p_T, p_Vote, on_plane='xy') :
	print(len(points_original))
	ids = np.random.choice(np.arange(len(points_original)), size=100, replace=False)
	points = points_original[ids].reshape(-1, 3)

	if on_plane == 'xy':
		p_Vote = p_Vote[:, :3]
	else :
		p_Vote = p_Vote[:, 3:]

	points_line = points + p_Vote[ids]
	vote_points = np.concatenate([points, points_line], axis=0)
	vote_lines = [[i, i + len(points)] for i in range(len(points))]
	vote_color = [[1, 0, 0] for i in range(len(vote_lines))]

	gt_green_v = p_green_R * 0.03 + p_T
	gt_red_v = p_red_R * 0.03 + p_T
	pose_points = np.concatenate([p_T.reshape(1, -1), gt_green_v.reshape(1, -1), gt_red_v.reshape(1, -1)], axis=0)
	pose_lines = [[0, 1], [0, 2]]
	pose_color = [[0, 1, 0] for i in range(len(pose_lines))]

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


def generate_RT(R, T, sym):   #generate sRT mat
	bs = T.shape[0]
	res = torch.zeros([bs, 4, 4], dtype=torch.float).to(T.device)
	# generate from green and red vec
	for i in range(bs):
		pred_green_vec = R[0][i, ...]
		if sym != 0:
			pred_red_vec = R[1][i, ...]
			new_z, new_y = refine_zy_vec(pred_green_vec, pred_red_vec)
			p_R = get_rot_mat_zy(new_z.view(1, -1), new_y.view(1, -1))[0]
		else :
			new_z, new_x = get_x_vec(pred_green_vec)
			p_R = get_rot_mat_zy(new_z.view(1, -1), new_x.view(1, -1))[0]
		RT = np.identity(4)
		RT[:3, :3] = p_R.cpu().numpy()
		RT[:3, 3] = T[i, ...].cpu().numpy()
		res[i, :, :] = torch.from_numpy(RT).to(T.device)
	return res


def generate_RT_ransac(points, votes, R, T, sym, threshold=0.005, max_iter=100, verbose=False):   #generate sRT mat
	bs = points.shape[0]
	res = torch.zeros([bs, 4, 4], dtype=torch.float).to(points.device)
	# generate from green and red vec
	for i in range(bs):
		p_t = T[i]
		pc_now = points[i]
		pred_green_vec = R[0][i, ...]
		if sym != 0:
			pred_red_vec = R[1][i, ...]
			p_line_green = pc_now + votes[i][:, :3]
			p_line_red = pc_now + votes[i][:, 3:]
			ratio_green, line_green, inliers_green = estimate3DLine(p_line_green, threshold, max_iter, translation=p_t)
			ratio_red, line_red, inliers_red = estimate3DLine(p_line_red, threshold, max_iter, translation=p_t)
			if verbose:
				print(f'The inlier ratio of the green vector prediction is {ratio_green}')
				print(f'The inlier ratio of the red vector prediction is {ratio_red}')

			if torch.dot(line_green, pred_green_vec) < 0:
				line_green = -line_green
			if torch.dot(line_red, pred_red_vec) < 0:
				line_red = -line_red
			p_R = get_rot_mat_zy(line_green.view(1, -1), line_red.view(1, -1))[0]
			# p_R = get_rot_mat_zy(line_green.view(1, -1), pred_red_vec.view(1, -1))[0]
		else:
			p_line_green = pc_now + votes[i]
			ratio_green, line_green, inliers_green = estimate3DLine(p_line_green, 0.005, max_iter=100)
			if verbose:
				print(f'The inlier ratio of the green vector prediction is {ratio_green}')

			if torch.dot(line_green, pred_green_vec) < 0:
				line_green = -line_green
			new_z, new_y = get_x_vec(line_green)
			p_R = get_rot_mat_zy(new_z.view(1, -1), new_y.view(1, -1))[0]
		RT = np.identity(4)
		RT[:3, :3] = p_R.cpu().numpy()
		RT[:3, 3] = T[i, ...].cpu().numpy()
		res[i, :, :] = torch.from_numpy(RT).to(T.device)
	return res


def generate_RT_ransac2(points, votes, vote_pts1, vote_pts2, T, sym, threshold=0.005, max_iter=100, verbose=False):   #generate sRT mat
	bs = points.shape[0]
	res = torch.zeros([bs, 4, 4], dtype=torch.float).to(points.device)
	# generate from green and red vec
	for i in range(bs):
		p_t = T[i]
		pc_now = points[i]
		pred_green_vec = (pc_now + vote_pts1[i]).mean(0) - p_t
		pred_green_vec = pred_green_vec / (torch.norm(pred_green_vec) + 1e-6)
		if sym != 0:
			p_line_green = pc_now + votes[i][:, :3]
			p_line_red = pc_now + votes[i][:, 3:]
			ratio_green, line_green, inliers_green = estimate3DLine(p_line_green, threshold, max_iter, translation=p_t)
			ratio_red, line_red, inliers_red = estimate3DLine(p_line_red, threshold, max_iter, translation=p_t)
			if verbose:
				print(f'The inlier ratio of the green vector prediction is {ratio_green}')
				print(f'The inlier ratio of the red vector prediction is {ratio_red}')

			if torch.dot(line_green, pred_green_vec) < 0:
				line_green = -line_green

			if sym == 2:
				pred_red_vec = (pc_now + vote_pts2[i]).mean(0) - p_t
				pred_red_vec = pred_red_vec / (torch.norm(pred_red_vec) + 1e-6)
				if torch.dot(line_red, pred_red_vec) < 0:
					line_red = -line_red
			new_z, new_y = refine_zy_vec(line_green, line_red)
			p_R = get_rot_mat_zy(new_z.view(1, -1), new_y.view(1, -1))[0]
			# p_R = get_rot_mat_zy(line_green.view(1, -1), pred_red_vec.view(1, -1))[0]
			# p_R = get_rot_mat_zy(pred_green_vec.view(1, -1), line_red.view(1, -1))[0]
		else:
			p_line_green = pc_now + votes[i]
			ratio_green, line_green, inliers_green = estimate3DLine(p_line_green, threshold, max_iter, translation=p_t)
			if verbose:
				print(f'The inlier ratio of the green vector prediction is {ratio_green}')

			if torch.dot(line_green, pred_green_vec) < 0:
				line_green = -line_green
			new_z, new_y = get_x_vec(line_green)
			p_R = get_rot_mat_zy(new_z.view(1, -1), new_y.view(1, -1))[0]
		RT = np.identity(4)
		RT[:3, :3] = p_R.cpu().numpy()
		RT[:3, 3] = T[i, ...].cpu().numpy()
		res[i, :, :] = torch.from_numpy(RT).to(T.device)
	return res


def generate_RT_ransac3(points, votes, vote_pts1, vote_pts2, T, sym):   #generate sRT mat
	bs = points.shape[0]
	res = torch.zeros([bs, 4, 4], dtype=torch.float).to(points.device)
	# generate from green and red vec
	for i in range(bs):
		p_t = T[i]
		pc_now = points[i]
		pred_green_vec = (pc_now + vote_pts1[i]).mean(0) - p_t
		pred_green_vec = pred_green_vec / (torch.norm(pred_green_vec) + 1e-6)
		if sym != 0:
			p_line_green = pc_now + votes[i][:, :3]
			p_line_red = pc_now + votes[i][:, 3:]
			line_green = linear_fitting_3D_points(p_line_green)
			line_red = linear_fitting_3D_points(p_line_red)

			if torch.dot(line_green, pred_green_vec) < 0:
				line_green = -line_green
			# if torch.dot(line_red, pred_red_vec) < 0:
			# 	line_red = -line_red
			if sym == 2:
				pred_red_vec = (pc_now + vote_pts2[i]).mean(0) - p_t
				pred_red_vec = pred_red_vec / (torch.norm(pred_red_vec) + 1e-6)
				if torch.dot(line_red, pred_red_vec) < 0:
					line_red = -line_red
			new_z, new_y = refine_zy_vec(line_green, line_red)
			p_R = get_rot_mat_zy(new_z.view(1, -1), new_y.view(1, -1))[0]
			# p_R = get_rot_mat_zy(line_green.view(1, -1), pred_red_vec.view(1, -1))[0]
			# p_R = get_rot_mat_zy(pred_green_vec.view(1, -1), line_red.view(1, -1))[0]
		else:
			p_line_green = pc_now + votes[i]
			line_green = linear_fitting_3D_points(p_line_green)

			if torch.dot(line_green, pred_green_vec) < 0:
				line_green = -line_green
			new_z, new_y = get_x_vec(line_green)
			p_R = get_rot_mat_zy(new_z.view(1, -1), new_y.view(1, -1))[0]
		RT = np.identity(4)
		RT[:3, :3] = p_R.cpu().numpy()
		RT[:3, 3] = T[i, ...].cpu().numpy()
		res[i, :, :] = torch.from_numpy(RT).to(T.device)
	return res


def get_plane(pc):
	# min least square
	n = pc.shape[0]
	A = torch.cat([pc[:, :2], torch.ones([n, 1], device=pc.device)], dim=-1)
	b = pc[:, 2].view(-1, 1)
	ATA = torch.mm(A.permute(1, 0), A)
	ATA_1 = torch.inverse(ATA)
	ATb = torch.mm(A.permute(1, 0), b)
	X = torch.mm(ATA_1, ATb)
	# return dn
	dn_up = torch.cat([X[0] * X[2], X[1] * X[2], -X[2]], dim=0)
	# dn_norm = X[0] * X[0] + X[1] * X[1] + 1.0
	dn = dn_up / torch.norm(dn_up)
	return dn


# def generate_RT_plane(points, vectors, R, T):
# 	bs = points.shape[0]
# 	res = torch.zeros([bs, 4, 4], dtype=torch.float).to(T.device)
# 	# generate from green and red vec
# 	for i in range(bs):
# 		points_now = points[i]
# 		pred_green = R[0][i, ...]
# 		pred_red = R[1][i, ...]
# 		points_xoy_now = points_now + vectors[i, :, :3]
# 		points_zox_now = points_now + vectors[i, :, 3:]
#
# 		pred_green_vote = get_plane(points_xoy_now)
# 		pred_red_vote = get_plane(points_zox_now)
# 		if torch.dot(pred_green_vote, pred_green) < 0:
# 			pred_green_vote = -pred_green_vote
# 		if torch.dot(pred_red_vote, pred_red) < 0:
# 			pred_red_vote = -pred_red_vote
#
# 		new_z, new_y = refine_zy_vec(pred_green_vote, pred_red_vote)
# 		p_R = get_rot_mat_zy(new_z.view(1, -1), new_y.view(1, -1))[0]
#
# 		RT = np.identity(4)
# 		RT[:3, :3] = p_R.cpu().numpy()
# 		RT[:3, 3] = T[i, ...].cpu().numpy()
# 		res[i, :, :] = torch.from_numpy(RT).to(T.device)
#
# 	return res


def generate_RT_plane(points, vectors, R, T):
	bs = points.shape[0]
	res = torch.zeros([bs, 4, 4], dtype=torch.float).to(T.device)
	# generate from green and red vec
	for i in range(bs):
		points_now = points[i]
		pred_green = R[0][i, ...]
		pred_red = R[1][i, ...]
		vectors_xoy_now = vectors[i, :, :3]
		vectors_zox_now = vectors[i, :, 3:]

		mask_green = torch.mm(vectors_xoy_now, pred_green.view(3, -1)).squeeze()
		vectors_xoy_plus = torch.index_select(vectors_xoy_now, 0, torch.where(mask_green > 0)[0])
		vectors_xoy_minus = torch.index_select(vectors_xoy_now, 0, torch.where(mask_green < 0)[0])
		vectors_xoy_now = torch.cat([vectors_xoy_plus, -vectors_xoy_minus], dim=0)
		vectors_xoy_now = vectors_xoy_now / (torch.norm(vectors_xoy_now, dim=1, keepdim=True) + 1e-6)

		mask_red = torch.mm(vectors_zox_now, pred_red.view(3, -1)).squeeze()
		vectors_zox_plus = torch.index_select(vectors_zox_now, 0, torch.where(mask_red > 0)[0])
		vectors_zox_minus = torch.index_select(vectors_zox_now, 0, torch.where(mask_red < 0)[0])
		vectors_zox_now = torch.cat([vectors_zox_plus, -vectors_zox_minus], dim=0)
		vectors_zox_now = vectors_zox_now / (torch.norm(vectors_zox_now, dim=1, keepdim=True) + 1e-6)

		pred_green_vote = torch.mean(vectors_xoy_now, dim=0)
		pred_red_vote = torch.mean(vectors_zox_now, dim=0)

		new_z, new_y = refine_zy_vec(pred_green_vote, pred_red_vote)
		p_R = get_rot_mat_zy(new_z.view(1, -1), new_y.view(1, -1))[0]

		RT = np.identity(4)
		RT[:3, :3] = p_R.cpu().numpy()
		RT[:3, 3] = T[i, ...].cpu().numpy()
		res[i, :, :] = torch.from_numpy(RT).to(T.device)

	return res


class PoseEstimationPredictor:
	def __init__(self, cfg):
		self.cfg = cfg
		# if cfg.sym_type == 0:
		# 	self.model = GSCPose_Re(cfg)
		# else:
		# 	self.model = GSCPose(cfg)
		self.model = GSCPose(cfg)
		total_params = sum(p.numel() for p in self.model.parameters())
		print("Total parameters:", total_params)
		self.sym_type = cfg.sym_type
		self.input_channel = cfg.input_channel

		# val_dataset = PoseDataset(cfg, phase='val')
		# self.model_size = val_dataset.model_size

		checkpoint = torch.load(f'{cfg.exp_dir}/gscpose_best_val.pth.tar')
		print('Pose Estimation: loading model checkpoint from epoch {}'.format(checkpoint['epoch']))
		self.model.load_state_dict(checkpoint['state_dict'])
		self.model.cuda().eval()

	def sample_points(self, cloud_xyz):
		replace = len(cloud_xyz) < self.cfg.num_pts
		ids = np.random.choice(np.arange(len(cloud_xyz)), size=self.cfg.num_pts, replace=replace)
		cloud_xyz = cloud_xyz[ids].reshape(-1, 3)
		return cloud_xyz, ids

		# pcd = toOpen3dCloud(cloud_xyz)
		# dwn_pcd = pcd.farthest_point_down_sample(num_samples=self.cfg.num_pts)
		# cloud_xyz_dwn = np.asarray(dwn_pcd.points).reshape(-1, 3)
		# return cloud_xyz_dwn.astype(np.float32)

	def predict(self, data):
		"""
		:param cloud_xyz: [bs, n_pts, 3]
		:return:
		"""
		with torch.no_grad():
			cloud_xyz = data['points']
			points = torch.as_tensor(cloud_xyz.astype(np.float32)).contiguous()
			points_cuda = points.cuda()
			s_time = time.time()
			output_dict = self.model(points_cuda, gt_R=None, gt_t=None, gt_size=None)
			d_time = time.time() - s_time
			print(f'Inference time: {d_time}')
			# p_green_R_vec = output_dict['p_green_R'].detach()
			p_T = output_dict['Pred_T'].detach()
			p_vote = output_dict['vote'].detach()
			p_vote_point1 = output_dict['vote_point1'].detach()
			p_vote_point2 = None
			if self.sym_type == 2:
				p_vote_point2 = output_dict['vote_point2'].detach()

			# if self.sym_type != 0:
			# 	# p_red_R_vec = output_dict['p_red_R'].detach()
			# 	# res = generate_RT([p_green_R_vec, p_red_R_vec], p_T, self.sym_type)
			#
			# 	# res = generate_RT_ransac(points_cuda, p_vote, [p_green_R_vec, p_red_R_vec], p_T, self.sym_type,
			# 	#                          threshold=0.008, max_iter=100, verbose=False)
			# 	res = generate_RT_ransac2(points_cuda, p_vote, p_vote_point1, p_vote_point2, p_T, self.sym_type,
			#                              threshold=0.01, max_iter=100, verbose=False)
			# 	# res = generate_RT_ransac3(points_cuda, p_vote, p_vote_point1, p_T, self.sym_type)
			# 	# res = generate_RT_plane(points_cuda, p_vote, [p_green_R_vec, p_red_R_vec], p_T)
			# else:
			# 	res = generate_RT([p_green_R_vec], p_T, self.sym_type)
			# 	# res = generate_RT_ransac([p_green_R_vec], p_T, self.sym_type)
			s_time = time.time()
			res = generate_RT_ransac2(points_cuda, p_vote, p_vote_point1, p_vote_point2, p_T, self.sym_type,
			                          threshold=0.005, max_iter=100, verbose=False)
			d_time = time.time() - s_time
			print(f'RANSAC time: {d_time}')
			# res = generate_RT_ransac3(points_cuda, p_vote, p_vote_point1, p_vote_point2, p_T, self.sym_type)

			return res.cpu().numpy(), p_vote.cpu().numpy()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='Sileance_Dataset', help='indicate dataset name')
	parser.add_argument('--obj_name', type=str, default='gear', help='indicate object name')
	args = parser.parse_args()
	dataset = args.dataset
	obj_name = args.obj_name

	from PoseEstimation.util.config import get_parser as get_pose_cfg
	cfg_file = f'{code_dir}/PoseEstimation/exp/{dataset}/{obj_name}/config_pose.yaml'
	cfg = get_pose_cfg(cfg_file)

	predictor = PoseEstimationPredictor(cfg)

	val_dataset = PoseDataset(cfg, phase='val')
	idx = random.randrange(0, len(val_dataset))
	print(idx)
	data = val_dataset[17]
	# data = val_dataset[1989]
	points = data['points'].numpy()
	gt_R = data['Rs'].numpy()
	gt_t = data['ts'].numpy()

	input = {
		'points': points[None]
	}

	_, _ = predictor.predict(input)
	_, _ = predictor.predict(input)
	_, _ = predictor.predict(input)

	# start_time = time.time()
	p_T, p_vote = predictor.predict(input)
	# duration = time.time() - start_time
	# print(f'Inference time: {duration}')

	# print('GT translation: ', gt_t)
	# print('Pred translation: ', p_T[0, :3, 3])
	# print(np.linalg.norm(gt_t-p_T[0, :3, 3]))

	if predictor.sym_type == 0:
		visualization_line(points, p_T[0, :3, 2], p_T[0, :3, 0], p_T[0, :3, 3], p_vote[0])
	else:
		visualization_plane(points, p_T[0, :3, 2], p_T[0, :3, 0], p_T[0, :3, 3], p_vote[0], on_plane='xy')
