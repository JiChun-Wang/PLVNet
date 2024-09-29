import copy
import os
import random
code_dir = os.path.dirname(os.path.realpath(__file__))
import glob
import json

import numpy as np
import cv2
import open3d as o3d
from scipy.spatial import cKDTree
import torch
from transforms3d.quaternions import quat2mat

from predictor_pointgroup import PointGroupPredictor
from predictor_gscpose import PoseEstimationPredictor
print(code_dir)
from util.utils import toOpen3dCloud, correct_pcd_normal_direction, cloud_sample, parse_camera_config
from PointGroup.util.config import get_parser as get_seg_cfg
from PoseEstimation.util.config import get_parser as get_pose_cfg
from util.DataReader import DataReader


random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


def find_nearest_points(src_xyz, query):
	kdtree = cKDTree(src_xyz)
	dists, indices = kdtree.query(query)
	query_xyz = src_xyz[indices]
	return query_xyz, indices


if __name__ == '__main__':
	obj_name_lst = ['IPARingScrew']
	dataset = 'IPA_Dataset'
	for obj_name in obj_name_lst:
		cfg_file = f'{code_dir}/PointGroup/exp/{dataset}/{obj_name}/config_pointgroup.yaml'
		cfg_seg = get_seg_cfg(cfg_file)
		cfg_file = f'{code_dir}/PoseEstimation/exp/{dataset}/{obj_name}/config_pose.yaml'
		cfg_pose = get_pose_cfg(cfg_file)

		data_dir = f'{code_dir}/data/{dataset}/{obj_name}'
		assert os.path.exists(data_dir), 'The target object does not exist!'

		cam_para_path = os.path.join(data_dir, 'parameter.json')
		datareader = DataReader(cam_para_path)

		cam_para_pth = os.path.join(data_dir, 'camera_params.txt')
		cam_params = parse_camera_config(cam_para_pth)

		K = np.array([[cam_params['fu'], 0., cam_params['cu']],
		              [0., cam_params['fv'], cam_params['cv']],
		              [0., 0., 1.]])

		clip_start = cam_params['clip_start']
		clip_end = cam_params['clip_end']
		quat = cam_params['rotation']
		tra = cam_params['location']
		cam_in_world = np.eye(4)
		cam_in_world[:3, :3] = quat2mat(quat)
		cam_in_world[:3, 3] = tra

		mesh_pcd = o3d.io.read_point_cloud(f'{data_dir}/mesh.pcd')

		seg_predictor = PointGroupPredictor(cfg_seg)
		pose_predictor = PoseEstimationPredictor(cfg_pose)

		# result_folder = f'{data_dir}/pred'
		# if os.path.exists(result_folder):
		# 	os.system(f'rm -rf {result_folder}')
		# os.makedirs(result_folder, exist_ok=False)
		# result_pp_folder = f'{data_dir}/pred_pp'
		# if os.path.exists(result_pp_folder):
		# 	os.system(f'rm -rf {result_pp_folder}')
		# os.makedirs(result_pp_folder, exist_ok=False)
		#
		# dpt_files = sorted(glob.glob(f"{data_dir}/depth/*.PNG"))
		# for idx_cycle in range(10):
		# 	cycle_result_folder = os.path.join(result_folder, f'cycle_{idx_cycle:04d}')
		# 	if not os.path.exists(cycle_result_folder):
		# 		os.makedirs(cycle_result_folder)
		#
		# 	cycle_result_pp_folder = os.path.join(result_pp_folder, f'cycle_{idx_cycle:04d}')
		# 	if not os.path.exists(cycle_result_pp_folder):
		# 		os.makedirs(cycle_result_pp_folder)
		#
		# 	cycle_dpt_files = sorted(glob.glob(f"{data_dir}/p_depth_real_world/cycle_{idx_cycle:04d}/*_depth_uint16.png"))
		#
		# 	name = int(os.path.basename(cycle_dpt_files[0]).split('_')[0])
		# 	result_file = os.path.join(cycle_result_folder, f'{name:03d}.json')
		# 	with open(result_file, 'w') as f:
		# 		json.dump([], f)
		# 	result_pp_file = os.path.join(cycle_result_pp_folder, f'{name:03d}.json')
		# 	with open(result_pp_file, 'w') as f:
		# 		json.dump([], f)
		#
		# 	for i, dpt_file in enumerate(cycle_dpt_files[1:], start=1):
		# 		print(f'Scene: {dpt_file}')
		# 		depth = datareader.read_depth_map(dpt_file)
		# 		dpt_back_file = cycle_dpt_files[0]
		# 		depth_back = datareader.read_depth_map(dpt_back_file)
		#
		# 		# fill hole
		# 		depth_back[depth_back<1.26] = 0
		# 		depth[depth<1.26] = 0
		# 		mask = np.abs(depth - depth_back) <= 0.005
		# 		depth[mask] = 0
		#
		# 		depth[depth<0.1] = 0
		#
		# 		xyz_map = datareader.depth2xyzmap(depth)
		# 		valid_mask = xyz_map[:, :, 2] >= 0.1
		# 		cloud_xyz_original = xyz_map[valid_mask].reshape(-1, 3)
		#
		# 		cloud_xyz_001 = cloud_sample(cloud_xyz_original, downsample_size=cfg_pose.ds_size)
		# 		cloud_xyz_005 = cloud_sample(cloud_xyz_001, downsample_size=cfg_seg.ds_size)
		# 		input = {'cloud_xyz': cloud_xyz_005}
		# 		sem_labels_005, ins_labels_fg = seg_predictor.predict(input, idx=i+idx_cycle*28)
		#
		# 		pred_lst = []
		# 		pred_pp_lst = []
		# 		if np.sum(sem_labels_005==1) > 0:
		# 			ins_labels_005 = np.ones(len(cloud_xyz_005)) * -100
		# 			ins_labels_005[sem_labels_005==1] = ins_labels_fg
		# 			_, indices = find_nearest_points(cloud_xyz_005, cloud_xyz_001)
		# 			ins_labels_001 = ins_labels_005[indices]
		#
		# 			ins_ids = np.unique(ins_labels_001)
		# 			input = {'points': []}
		# 			for ins_id in ins_ids:
		# 				if ins_id == -100:
		# 					continue
		# 				ins_mask = ins_labels_001==ins_id
		# 				n_pt = np.sum(ins_mask)
		# 				print(f'{ins_id}: {n_pt}')
		# 				if n_pt < 2048:
		# 					print(f"segment too small, n_pt={n_pt}")
		# 					continue
		#
		# 				cloud_xyz_ins = cloud_xyz_001[ins_mask]
		# 				cloud_xyz_ins = pose_predictor.sample_points(cloud_xyz_ins)
		# 				input['points'].append(cloud_xyz_ins)
		#
		# 			if len(input['points']) != 0:
		# 				input['points'] = np.asarray(input['points'])
		#
		# 				p_RT, p_Vote = pose_predictor.predict(input)
		#
		# 				for k in range(len(input['points'])):
		# 					ob_in_cam = np.eye(4)
		# 					ob_in_cam[:3, :3] = p_RT[k][:3, :3]
		# 					ob_in_cam[:3, 3] = p_RT[k][:3, 3]
		#
		# 					target = copy.deepcopy(mesh_pcd)
		# 					points_in_cam = copy.deepcopy(np.asarray(input['points'][k]))
		# 					source = toOpen3dCloud(points_in_cam)
		# 					trans_init = np.linalg.inv(ob_in_cam)
		# 					reg_p2p = o3d.pipelines.registration.registration_icp(
		# 						source, target, 0.02, trans_init,
		# 						o3d.pipelines.registration.TransformationEstimationPointToPoint())
		# 					ob_in_cam_pp = np.linalg.inv(reg_p2p.transformation)
		#
		# 					pred_lst.append({
		# 						'R': ob_in_cam[:3, :3].tolist(),
		# 						'score': 1.0,
		# 						't': ob_in_cam[:3, 3].tolist()
		# 					})
		#
		# 					pred_pp_lst.append({
		# 						'R': ob_in_cam_pp[:3, :3].tolist(),
		# 						'score': 1.0,
		# 						't': ob_in_cam_pp[:3, 3].tolist()
		# 					})
		#
		# 		name = int(os.path.basename(cycle_dpt_files[i]).split('_')[0])
		# 		result_file = os.path.join(cycle_result_folder, f'{name:03d}.json')
		# 		with open(result_file, 'w') as f:
		# 			json.dump(pred_lst, f)
		# 		result_pp_file = os.path.join(cycle_result_pp_folder, f'{name:03d}.json')
		# 		with open(result_pp_file, 'w') as f:
		# 			json.dump(pred_pp_lst, f)

		pcd = o3d.io.read_point_cloud(f"{code_dir}/real_world_scene.ply")
		cloud_xyz_original = np.float32(pcd.points)
		cloud_xyz_001, _ = cloud_sample(cloud_xyz_original, downsample_size=cfg_pose.ds_size)
		cloud_xyz_005, _ = cloud_sample(cloud_xyz_001, downsample_size=cfg_seg.ds_size)
		input = {'cloud_xyz': cloud_xyz_005}
		sem_labels_005, ins_labels_fg = seg_predictor.predict(input, idx=0)

		pred_pp_lst = []
		if np.sum(sem_labels_005 == 1) > 0:
			ins_labels_005 = np.ones(len(cloud_xyz_005)) * -100
			ins_labels_005[sem_labels_005 == 1] = ins_labels_fg
			_, indices = find_nearest_points(cloud_xyz_005, cloud_xyz_001)
			ins_labels_001 = ins_labels_005[indices]

			ins_ids = np.unique(ins_labels_001)
			input = {'points': []}
			for ins_id in ins_ids:
				if ins_id == -100:
					continue
				ins_mask = ins_labels_001 == ins_id
				n_pt = np.sum(ins_mask)
				print(f'{ins_id}: {n_pt}')
				if n_pt < 2048:
					print(f"segment too small, n_pt={n_pt}")
					continue

				cloud_xyz_ins = cloud_xyz_001[ins_mask]
				cloud_xyz_ins, _ = pose_predictor.sample_points(cloud_xyz_ins)
				input['points'].append(cloud_xyz_ins)

			if len(input['points']) != 0:
				input['points'] = np.asarray(input['points'])
				p_RT, p_Vote = pose_predictor.predict(input)

				for k in range(len(input['points'])):
					ob_in_cam = np.eye(4)
					ob_in_cam[:3, :3] = p_RT[k][:3, :3]
					ob_in_cam[:3, 3] = p_RT[k][:3, 3]
					ob_in_world = cam_in_world @ ob_in_cam

					target = copy.deepcopy(mesh_pcd)
					points_in_cam = copy.deepcopy(np.asarray(input['points'][k]))
					source = toOpen3dCloud(points_in_cam)

					trans_init = np.linalg.inv(ob_in_cam)
					reg_p2p = o3d.pipelines.registration.registration_icp(
						source, target, 0.02, trans_init,
						o3d.pipelines.registration.TransformationEstimationPointToPoint())
					ob_in_cam_pp = np.linalg.inv(reg_p2p.transformation)
					ob_in_world_pp = cam_in_world @ ob_in_cam_pp

					pred_pp_lst.append({
						'R': ob_in_world_pp[:3, :3].tolist(),
						'score': 1.0,
						't': ob_in_world_pp[:3, 3].tolist()
					})

		result_pp_file = os.path.join(code_dir, f'real_world_scene.json')
		with open(result_pp_file, 'w') as f:
			json.dump(pred_pp_lst, f)

