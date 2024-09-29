import os, glob, pickle, copy
code_dir = os.path.dirname(os.path.realpath(__file__))
import math
import argparse
import random

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import open3d as o3d

from PoseEstimation.network.CatPose import CatGSCPose
from PoseEstimation.util.utils import toOpen3dCloud
from util.utils import to_homo, toOpen3dCloud, depth2xyzmap, cloud_sample, find_nearest_points, kabsch
from util.eval_utils import compute_degree_cm_mAP
from predictor_pointgroup import PointGroupPredictor
from PointGroup.util.config import get_parser as get_seg_cfg
from PoseEstimation.util.config import get_parser as get_pose_cfg


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

	if on_plane == 'xy' :
		p_Vote = p_Vote[:, :3]
	else :
		p_Vote = p_Vote[:, 3 :]

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


def gettrans(p_green_vec, p_red_vec):
	sources = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32)
	targets = np.vstack([np.zeros(3), p_green_vec, p_red_vec])

	P = sources.T - sources.T.mean(1).reshape((3, 1))
	Q = targets.T - targets.T.mean(1).reshape((3, 1))
	R = kabsch(P.T, Q.T)
	return R


def generate_RT_fsnet(R, T, sym):   #generate sRT mat
	bs = T.shape[0]
	res = torch.zeros([bs, 4, 4], dtype=torch.float).to(T.device)
	# generate from green and red vec
	for i in range(bs):
		pred_green_vec = R[0][i, ...]
		if sym != 0:
			pred_red_vec = R[1][i, ...]
			p_R = gettrans(pred_green_vec.cpu().numpy(), pred_red_vec.cpu().numpy())
		else :
			new_z, new_x = get_x_vec(pred_green_vec)
			p_R = get_rot_mat_zy(new_z.view(1, -1), new_x.view(1, -1))[0]
		RT = np.identity(4)
		RT[:3, :3] = p_R
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

		self.model = CatGSCPose(cfg)
		self.sym_type = cfg.sym_type
		self.input_channel = cfg.input_channel

		checkpoint = torch.load(f'{cfg.exp_dir}/gscpose_best_train.pth.tar')
		print(f'{cfg.exp_dir}/gscpose_best_train.pth.tar')
		print('Pose Estimation: loading model checkpoint from epoch {}'.format(checkpoint['epoch']))
		self.model.load_state_dict(checkpoint['state_dict'])
		self.model.cuda().eval()

	def sample_points(self, cloud_xyz):
		replace = len(cloud_xyz) < self.cfg.num_pts
		ids = np.random.choice(np.arange(len(cloud_xyz)), size=self.cfg.num_pts, replace=replace)
		cloud_xyz = cloud_xyz[ids].reshape(-1, 3)
		return cloud_xyz

	def predict(self, cloud_xyz, mean_shape):
		"""
		:param cloud_xyz: [bs, n_pts, 3]
		:param mean_shape: [3,]
		:return:
		"""
		with torch.no_grad():
			points = torch.as_tensor(cloud_xyz.astype(np.float32)).contiguous()
			points_cuda = points.cuda()
			mean_shape = torch.as_tensor(mean_shape.astype(np.float32)).contiguous()
			mean_shape = mean_shape.cuda()
			output_dict = self.model(points_cuda, gt_R=None, gt_t=None, gt_s=None, mean_shape=None)

			p_green_R_vec = output_dict['p_green_R'].detach()
			p_T = output_dict['Pred_T'].detach()
			p_s = output_dict['Pred_s'].detach()
			pred_s = p_s + mean_shape

			if self.sym_type != 0:
				p_red_R_vec = output_dict['p_red_R'].detach()
				res = generate_RT([p_green_R_vec, p_red_R_vec], p_T, self.sym_type)
				# res = generate_RT_plane(points_cuda, p_vote, [p_green_R_vec, p_red_R_vec], p_T)
			else:
				res = generate_RT([p_green_R_vec], p_T, self.sym_type)

			return pred_s.cpu().numpy(), res.cpu().numpy()


def get_test_scale(c, scale, is_nocs=False):
	if c == 'hnm_ENG_CVM_CVM_T2050322201-000B':
		x = 83.9
		y = 34.0
		z = 36.0
	elif c == 'hnm_nist_waterproof_male':
		x = 63.2
		y = 32.5
		z = 35.4
	elif c == 'hnm_cylinder_ENG_CVM_CVM_T2111032201-007_A':
		x = 34.4
		y = 14.6
		z = 47.8
	elif c == 'hnm_ENG_CVM_CVM_T2050323201-000B':
		x = 64.0
		y = 34.0
		z = 36.5
	elif c == 'hnm_ENG_CVM_CVM_T2032162201-000B':
		x = 82.8
		y = 25.6
		z = 24.6
	elif c == 'hnm_cylinder_ENG_CVM_CVM_T2111020201-007_A':
		x = 34.4
		y = 14.6
		z = 48.4
	elif c == 'hnm_ENG_CVM_CVM_T2030242201-000B':
		x = 50.4
		y = 34.1
		z = 34.8
	elif c == 'hnm_ENG_CVM_CVM_T2050182201-000B':
		x = 64.0
		y = 34.0
		z = 36.0
	else:
		print('This category is not recorded in my little brain.')
		raise NotImplementedError

	scale = np.array([x, y, z]) * scale
	if is_nocs:
		diameter = np.sqrt(np.linalg.norm(scale))
		return np.array([diameter, diameter, diameter]) / 1000.
	return scale / 1000.


def get_meta_info(meta, is_nocs=False):
	ids = [i for i in meta['visibility_rate'].keys()]
	detection_dict = {
		'gt_class_ids': [],
		'gt_scales': [],
		'gt_RTs': []
	}

	cam_in_world = meta['cam_in_world']
	for _, id in enumerate(ids):
		if meta['visibility_rate'][id] >= 0.75:
			detection_dict['gt_class_ids'].append(1)
			detection_dict['gt_scales'].append(get_test_scale(meta['model'], meta['scale'], is_nocs))
			detection_dict['gt_RTs'].append(np.linalg.inv(cam_in_world)@meta['poses'][id])

	detection_dict['gt_class_ids'] = np.asarray(detection_dict['gt_class_ids'])
	detection_dict['gt_scales'] = np.asarray(detection_dict['gt_scales'])
	detection_dict['gt_RTs'] = np.asarray(detection_dict['gt_RTs'])

	return detection_dict


if __name__ == '__main__':
	random.seed(1234)
	np.random.seed(1234)
	torch.manual_seed(1234)
	torch.cuda.manual_seed_all(1234)

	mean_shape = np.array([57.900, 27.065, 35.946]) * 1.75 / 1000.
	dataset = 'CatBP_Dataset'
	obj_name = 'all_wo_con_l1'
	start_cycle = 41
	end_cycle = 46
	num_per_cycle = 30

	cfg_file = f'{code_dir}/PointGroup/exp/{dataset}/all/config_pointgroup.yaml'
	cfg_seg = get_seg_cfg(cfg_file)
	cfg_file = f'{code_dir}/PoseEstimation/exp/{dataset}/{obj_name}/config_pose.yaml'
	cfg_pose = get_pose_cfg(cfg_file)

	data_dir = f'{code_dir}/data/{dataset}/all'
	assert os.path.exists(data_dir), 'The target object does not exist!'

	# cam_para_path = os.path.join(data_dir, 'parameter.json')
	# datareader = DataReader(cam_para_path)

	# mesh_pcd = o3d.io.read_point_cloud(f'{data_dir}/mesh.pcd')

	seg_predictor = PointGroupPredictor(cfg_seg)
	pose_predictor = PoseEstimationPredictor(cfg_pose)

	result_folder = f'{data_dir}/pred/{obj_name}_cycles{start_cycle}-{end_cycle}({num_per_cycle})'
	if not os.path.exists(result_folder):
		os.makedirs(result_folder, exist_ok=True)

	pred_results = []
	for idx_cycle in range(start_cycle, end_cycle):
		cycle_dpt_files = sorted(glob.glob(f"{data_dir}/p_depth/cycle_{idx_cycle:04d}/*_depth.png"))[:num_per_cycle]
		cycle_seg_files = sorted(glob.glob(f"{data_dir}/p_segmentation/cycle_{idx_cycle:04d}/*_segmentation.png"))[:num_per_cycle]
		cycle_gt_files = sorted(glob.glob(f"{data_dir}/gt/cycle_{idx_cycle:04d}/*.pkl"))[:num_per_cycle]

		for i, dpt_file in enumerate(cycle_dpt_files):
			print(f'Scene: {dpt_file}')
			dpt_map = cv2.imread(dpt_file, -1) / 1e3
			seg_map = cv2.imread(cycle_seg_files[i], -1).astype(int)
			with open(cycle_gt_files[i], 'rb') as ff:
				meta = pickle.load(ff)

			if len(meta['poses']) == 0:
				continue
			detection_dict = get_meta_info(meta, is_nocs=obj_name=='nocs')

			xyz_map = depth2xyzmap(dpt_map, meta['K'])
			valid_mask = (seg_map != 0) & (seg_map != 1)
			cloud_xyz_original = xyz_map[valid_mask].reshape(-1, 3)

			cloud_xyz_001 = cloud_xyz_original.copy()
			cloud_xyz_005, _ = cloud_sample(cloud_xyz_001, downsample_size=cfg_seg.ds_size)
			seg_input = {'cloud_xyz': cloud_xyz_005}
			sem_labels_005, ins_labels_fg = seg_predictor.predict(seg_input, idx=i + idx_cycle * num_per_cycle)

			pred_lst = []
			if np.sum(sem_labels_005 == 1) > 0:
				ins_labels_005 = np.ones(len(cloud_xyz_005)) * -100
				ins_labels_005[sem_labels_005 == 1] = ins_labels_fg
				_, indices = find_nearest_points(cloud_xyz_005, cloud_xyz_001)
				ins_labels_001 = ins_labels_005[indices]

				input = []
				ins_ids = np.unique(ins_labels_001)
				for ins_id in ins_ids:
					if ins_id == -100:
						continue
					ins_mask = ins_labels_001 == ins_id
					n_pt = np.sum(ins_mask)
					print(f'{ins_id}: {n_pt}')
					if n_pt < 512:
						print(f"segment too small, n_pt={n_pt}")
						continue

					cloud_xyz_ins = cloud_xyz_001[ins_mask]
					cloud_xyz_ins = pose_predictor.sample_points(cloud_xyz_ins)
					input.append(cloud_xyz_ins)

				if len(input) != 0:
					pred_scales, pred_RTs = pose_predictor.predict(np.asarray(input), mean_shape)

					p_RTs = []
					item = meta['model']
					mesh_file = f'{code_dir}/data/{dataset}/all/model/{item}/{item}.obj'
					mesh = o3d.io.read_triangle_mesh(mesh_file)
					model_pcd = o3d.geometry.PointCloud()
					if item == 'hnm_ENG_CVM_CVM_T2032162201-000B':
						model_pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * 0.75 * 1.5)
					else:
						model_pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * 1.5)
					for k in range(len(input)):
						ob_in_cam = np.eye(4)
						ob_in_cam[:3, :3] = pred_RTs[k][:3, :3]
						ob_in_cam[:3, 3] = pred_RTs[k][:3, 3]

						target = copy.deepcopy(model_pcd)
						points_in_cam = copy.deepcopy(np.asarray(input[k]))
						source = toOpen3dCloud(points_in_cam)

						trans_init = np.linalg.inv(ob_in_cam)
						reg_p2p = o3d.pipelines.registration.registration_icp(
							source, target, 0.02, trans_init,
							o3d.pipelines.registration.TransformationEstimationPointToPoint())
						ob_in_cam_pp = np.linalg.inv(reg_p2p.transformation)
						p_RTs.append(ob_in_cam_pp)
					pred_RTs = np.asarray(p_RTs)

					detection_dict['pred_RTs'] = pred_RTs
					detection_dict['pred_scales'] = pred_scales
					detection_dict['pred_class_ids'] = np.ones((len(input),), dtype=np.int32)
					detection_dict['pred_scores'] = np.ones((len(input),)) * 0.9
					pred_results.append(detection_dict)
				else:
					detection_dict['pred_RTs'] = np.zeros((0, 4, 4))
					detection_dict['pred_scales'] = np.zeros((0, 4, 4))
					pred_results.append(detection_dict)

	pred_result_save_path = f'{result_folder}/pred_result.pkl'
	with open(pred_result_save_path, 'wb') as file:
		pickle.dump(pred_results, file)

	degree_thres_list = list(range(0, 61, 1))
	shift_thres_list = [i / 2 for i in range(21)]
	iou_thres_list = [i / 100 for i in range(101)]

	# iou_aps, pose_aps, iou_acc, pose_acc = compute_mAP(pred_results, output_path, degree_thres_list, shift_thres_list,
	#                                                  iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True,)
	synset_names = ['BG'] + ['hmn']
	iou_aps, pose_aps = compute_degree_cm_mAP(pred_results, synset_names, result_folder, degree_thres_list,
	                                          shift_thres_list,
	                                          iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True, )

	# # fw = open('{0}/eval_logs.txt'.format(result_dir), 'a')
	iou_25_idx = iou_thres_list.index(0.25)
	iou_50_idx = iou_thres_list.index(0.5)
	iou_75_idx = iou_thres_list.index(0.75)
	degree_05_idx = degree_thres_list.index(5)
	degree_10_idx = degree_thres_list.index(10)
	shift_01_idx = shift_thres_list.index(1)
	shift_02_idx = shift_thres_list.index(2)
	shift_05_idx = shift_thres_list.index(5)

	print('average mAP:')
	print('3D IoU at 25: {:.1f}'.format(iou_aps[-1, iou_25_idx] * 100))
	print('3D IoU at 50: {:.1f}'.format(iou_aps[-1, iou_50_idx] * 100))
	print('3D IoU at 75: {:.1f}'.format(iou_aps[-1, iou_75_idx] * 100))
	print('5 degree, 1cm: {:.1f}'.format(pose_aps[-1, degree_05_idx, shift_01_idx] * 100))
	print('5 degree, 2cm: {:.1f}'.format(pose_aps[-1, degree_05_idx, shift_02_idx] * 100))
	print('5 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_05_idx, shift_05_idx] * 100))
	print('10 degree, 1cm: {:.1f}'.format(pose_aps[-1, degree_10_idx, shift_01_idx] * 100))
	print('10 degree, 2cm: {:.1f}'.format(pose_aps[-1, degree_10_idx, shift_02_idx] * 100))
	print('10 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_10_idx, shift_05_idx] * 100))

	for idx in range(1, len(synset_names)):
		print('category {}'.format(synset_names[idx]))
		print('mAP:')
		print('3D IoU at 25: {:.1f}'.format(iou_aps[idx, iou_25_idx] * 100))
		print('3D IoU at 50: {:.1f}'.format(iou_aps[idx, iou_50_idx] * 100))
		print('3D IoU at 75: {:.1f}'.format(iou_aps[idx, iou_75_idx] * 100))
		print('5 degree, 1cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_01_idx] * 100))
		print('5 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_02_idx] * 100))
		print('5 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_05_idx] * 100))
		print('10 degree, 1cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_01_idx] * 100))
		print('10 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_02_idx] * 100))
		print('10 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_05_idx] * 100))

