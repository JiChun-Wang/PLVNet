import copy
import os, time
import random
code_dir = os.path.dirname(os.path.realpath(__file__))
import glob
import json

import numpy as np
import cv2
import open3d as o3d
from scipy.spatial import cKDTree
from transforms3d.quaternions import quat2mat
import torch

from predictor_pointgroup import PointGroupPredictor
from predictor_gscpose import PoseEstimationPredictor, visualization_line, visualization_plane
print(code_dir)
from util.utils import toOpen3dCloud, depth2xyzmap, correct_pcd_normal_direction, depth2xyzmap_w_offset, cloud_sample, defor_2D
from PointGroup.util.config import get_parser as get_seg_cfg
from PoseEstimation.util.config import get_parser as get_pose_cfg


random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


outlier_dict = {
	"bunny": 1.51,
    "candlestick": 1.01,
    "pepper": 1.51,
    "brick": 1.01,
    "gear": 2.01,
    "tless_20": 1.01,
    "tless_22": 1.01,
    "tless_29": 1.01
}

num_scenes_dict = {
	"bunny": 81,
    "candlestick": 61,
    "pepper": 91,
    "brick": 151,
    "gear": 61,
    "tless_20": 100,
    "tless_22": 101,
    "tless_29": 80
}


def parse_camera_config(file):
	with open(file) as f:
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


def read_depth_map(path, clip_start, clip_end):
	depth = cv2.imread(path, -1) / 65535.0
	depth = clip_start + (clip_end - clip_start) * depth
	return depth


def fill_depth_normal(cloud_xyz):
	pcd = toOpen3dCloud(cloud_xyz)
	pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.003, max_nn=30))
	pcd = correct_pcd_normal_direction(pcd)
	normals = np.asarray(pcd.normals).copy()
	return normals.astype(np.float32)


def find_nearest_points(src_xyz, query):
	kdtree = cKDTree(src_xyz)
	dists, indices = kdtree.query(query)
	query_xyz = src_xyz[indices]
	return query_xyz, indices


def point_cloud_filtering(points, normals, k=20, u=2.0, debug=False):
	pcd = toOpen3dCloud(points, normals=normals)
	sor_pcd, idx = pcd.remove_statistical_outlier(k, u)
	if debug:
		sor_pcd.paint_uniform_color([0, 0, 1])
		sor_noise_pcd = pcd.select_by_index(idx, invert=True)
		sor_noise_pcd.paint_uniform_color([1, 0, 0])
		o3d.visualization.draw_geometries([sor_pcd,sor_noise_pcd], window_name="SOR")
	points = np.asarray(sor_pcd.points)
	normals = np.asarray(sor_pcd.normals)
	return points, normals


if __name__ == '__main__':
	# obj_name_lst = ['bunny', 'candlestick', 'pepper', 'brick','gear', 'tless_20', 'tless_22', 'tless_29']
	obj_name_lst = ['gear']
	dataset = 'Sileance_Dataset'
	for obj_name in obj_name_lst:
		cfg_file = f'{code_dir}/PointGroup/exp/{dataset}/{obj_name}/config_pointgroup.yaml'
		cfg_seg = get_seg_cfg(cfg_file)
		cfg_file = f'{code_dir}/PoseEstimation/exp/{dataset}/{obj_name}/config_pose.yaml'
		cfg_pose = get_pose_cfg(cfg_file)

		data_dir = f'{code_dir}/data/{dataset}/{obj_name}'
		assert os.path.exists(data_dir), 'The target object does not exist!'

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

		result_folder = f'{data_dir}/pred'
		if os.path.exists(result_folder):
			os.system(f'rm -rf {result_folder}')
		os.makedirs(result_folder, exist_ok=False)
		result_pp_folder = f'{data_dir}/pred_pp'
		if os.path.exists(result_pp_folder):
			os.system(f'rm -rf {result_pp_folder}')
		os.makedirs(result_pp_folder, exist_ok=False)

		dpt_files = sorted(glob.glob(f"{data_dir}/depth/*.PNG"))
		if obj_name in ['candlestick', 'tless_20', 'tless_22', 'tless_29']:
			dpt_files_cycle_1 = dpt_files[::2]
			dpt_files_cycle_2 = dpt_files[1:][::2]
			dpt_files = dpt_files_cycle_1 + dpt_files_cycle_2
		for idx_cycle in range(2 if obj_name != 'bunny' else 4):
			num_scenes = num_scenes_dict[obj_name]
			start_scene_idx = idx_cycle*num_scenes
			end_scene_idx = idx_cycle*num_scenes+num_scenes
			cycle_dpt_files = dpt_files[start_scene_idx:end_scene_idx]

			# dpt_file = cycle_dpt_files[10]
			#
			# depth = read_depth_map(dpt_file, clip_start, clip_end)
			# dpt_back_file = cycle_dpt_files[0]
			# depth_back = read_depth_map(dpt_back_file, clip_start, clip_end)
			#
			# # fill hole
			# depth_back[depth_back > outlier_dict[obj_name]] = 0
			# depth[depth > outlier_dict[obj_name]] = 0
			# mask = np.abs(depth - depth_back) <= 0.005
			# depth[mask] = 0
			#
			# depth[depth < 0.1] = 0
			# depth[depth > outlier_dict[obj_name]] = 0
			#
			# xyz_map = depth2xyzmap(depth, K)
			# if obj_name in ['gear']:
			# 	xyz_map = xyz_map[::2, ::2]
			# valid_mask = xyz_map[:, :, 2] >= 0.1
			# cloud_xyz_original = xyz_map[valid_mask].reshape(-1, 3)
			#
			# cloud_xyz_001, indices = cloud_sample(cloud_xyz_original, downsample_size=cfg_pose.ds_size)
			# # cloud_xyz_001 = cloud_xyz_original.copy()
			# cloud_xyz_005, _ = cloud_sample(cloud_xyz_001, downsample_size=cfg_seg.ds_size)
			# input = {'cloud_xyz': cloud_xyz_005}
			# sem_labels_005, ins_labels_fg = seg_predictor.predict(input, idx=0)
			#
			# pred_lst = []
			# pred_pp_lst = []
			# if np.sum(sem_labels_005 == 1) > 0:
			# 	ins_labels_005 = np.ones(len(cloud_xyz_005)) * -100
			# 	ins_labels_005[sem_labels_005 == 1] = ins_labels_fg
			# 	_, indices = find_nearest_points(cloud_xyz_005, cloud_xyz_001)
			# 	ins_labels_001 = ins_labels_005[indices]
			#
			# 	ins_ids = np.unique(ins_labels_001)
			# 	input = {'points': []}
			# 	for ins_id in ins_ids:
			# 		if ins_id == -100:
			# 			continue
			# 		ins_mask = ins_labels_001 == ins_id
			# 		n_pt = np.sum(ins_mask)
			# 		print(f'{ins_id}: {n_pt}')
			# 		if n_pt < 2048:
			# 			print(f"segment too small, n_pt={n_pt}")
			# 			continue
			#
			# 		cloud_xyz_ins = cloud_xyz_001[ins_mask]
			# 		cloud_xyz_ins, ids = pose_predictor.sample_points(cloud_xyz_ins)
			# 		input['points'].append(cloud_xyz_ins)
			#
			# 	if len(input['points']) != 0:
			# 		input['points'] = np.asarray(input['points'])
			# 		p_RT, p_Vote = pose_predictor.predict(input)
			#
			# 		for k in range(len(input['points'])):
			# 			# ob_in_cam = np.eye(4)
			# 			# ob_in_cam[:3, :3] = p_RT[k][:3, :3]
			# 			# ob_in_cam[:3, 3] = p_RT[k][:3, 3]
			# 			# ob_in_world = cam_in_world @ ob_in_cam
			# 			#
			# 			# target = copy.deepcopy(mesh_pcd)
			# 			# points_in_cam = copy.deepcopy(np.asarray(input['points'][k]))
			# 			# source = toOpen3dCloud(points_in_cam)
			# 			#
			# 			# trans_init = np.linalg.inv(ob_in_cam)
			# 			# reg_p2p = o3d.pipelines.registration.registration_icp(
			# 			# 	source, target, 0.02, trans_init,
			# 			# 	o3d.pipelines.registration.TransformationEstimationPointToPoint())
			# 			# ob_in_cam_pp = np.linalg.inv(reg_p2p.transformation)
			# 			# ob_in_world_pp = cam_in_world @ ob_in_cam_pp
			#
			# 			visualization_plane(input['points'][k], p_RT[k, :3, 2], p_RT[k, :3, 0], p_RT[k, :3, 3], p_Vote[k],
			# 			                    on_plane='xy')
			#
			# 			# pred_lst.append({
			# 			# 	'R': ob_in_world[:3, :3].tolist(),
			# 			# 	'score': 1.0,
			# 			# 	't': ob_in_world[:3, 3].tolist()
			# 			# })
			# 			#
			# 			# pred_pp_lst.append({
			# 			# 	'R': ob_in_world_pp[:3, :3].tolist(),
			# 			# 	'score': 1.0,
			# 			# 	't': ob_in_world_pp[:3, 3].tolist()
			# 			# })
			#
			# break

			result_file = os.path.join(result_folder, os.path.basename(cycle_dpt_files[0]).split('.')[0] + '.json')
			with open(result_file, 'w') as f:
				json.dump([], f)

			result_pp_file = os.path.join(result_pp_folder, os.path.basename(cycle_dpt_files[0]).split('.')[0] + '.json')
			with open(result_pp_file, 'w') as f:
				json.dump([], f)

			for i, dpt_file in enumerate(cycle_dpt_files[1:], start=1):
				print(f'Scene: {dpt_file}')
				depth = read_depth_map(dpt_file, clip_start, clip_end)
				dpt_back_file = cycle_dpt_files[0]
				depth_back = read_depth_map(dpt_back_file, clip_start, clip_end)

				# fill hole
				depth_back[depth_back>outlier_dict[obj_name]] = 0
				depth[depth>outlier_dict[obj_name]] = 0
				mask = np.abs(depth - depth_back) <= 0.005
				depth[mask] = 0

				depth[depth<0.1] = 0
				depth[depth>outlier_dict[obj_name]] = 0

				xyz_map = depth2xyzmap(depth, K)
				if obj_name in ['gear']:
					xyz_map = xyz_map[::2, ::2]
				valid_mask = xyz_map[:, :, 2] >= 0.1
				cloud_xyz_original = xyz_map[valid_mask].reshape(-1, 3)

				cloud_xyz_001, indices = cloud_sample(cloud_xyz_original, downsample_size=cfg_pose.ds_size)
				# cloud_xyz_001 = cloud_xyz_original.copy()
				cloud_xyz_005, _ = cloud_sample(cloud_xyz_001, downsample_size=cfg_seg.ds_size)
				input = {'cloud_xyz': cloud_xyz_005}
				sem_labels_005, ins_labels_fg = seg_predictor.predict(input, idx=i+idx_cycle*num_scenes, vis_filter=True)

				pred_lst = []
				pred_pp_lst = []
				if np.sum(sem_labels_005==1) > 0:
					ins_labels_005 = np.ones(len(cloud_xyz_005)) * -100
					ins_labels_005[sem_labels_005==1] = ins_labels_fg
					_, indices = find_nearest_points(cloud_xyz_005, cloud_xyz_001)
					ins_labels_001 = ins_labels_005[indices]

					ins_ids = np.unique(ins_labels_001)
					input = {'points': []}
					for ins_id in ins_ids:
						if ins_id == -100:
							continue
						ins_mask = ins_labels_001==ins_id
						n_pt = np.sum(ins_mask)
						print(f'{ins_id}: {n_pt}')
						if n_pt < 2048:
							print(f"segment too small, n_pt={n_pt}")
							continue

						cloud_xyz_ins = cloud_xyz_001[ins_mask]
						cloud_xyz_ins, ids = pose_predictor.sample_points(cloud_xyz_ins)
						input['points'].append(cloud_xyz_ins)

					if len(input['points']) != 0:
						input['points'] = np.asarray(input['points'])
						s_time = time.time()
						p_RT, p_Vote = pose_predictor.predict(input)
						d_time = time.time() - s_time
						print(f'Inference time: {d_time}')

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

							pred_lst.append({
								'R': ob_in_world[:3, :3].tolist(),
								'score': 1.0,
								't': ob_in_world[:3, 3].tolist()
							})

							pred_pp_lst.append({
								'R': ob_in_world_pp[:3, :3].tolist(),
								'score': 1.0,
								't': ob_in_world_pp[:3, 3].tolist()
							})

				result_file = os.path.join(result_folder, os.path.basename(dpt_file).split('.')[0]+'.json')
				with open(result_file, 'w') as f:
					json.dump(pred_lst, f)

				result_file = os.path.join(result_pp_folder, os.path.basename(dpt_file).split('.')[0] + '.json')
				with open(result_file, 'w') as f:
					json.dump(pred_pp_lst, f)


		#### -----------------GT segmentation ------------------- #####
		# dpt_gt_files = sorted(glob.glob(f"{data_dir}/depth_gt/*.PNG"))
		# seg_files = sorted(glob.glob(f"{data_dir}/segmentation/*.PNG"))
		# gt_files = sorted(glob.glob(f"{data_dir}/gt/*.json"))
		# mesh_pcd = o3d.io.read_point_cloud(f'{data_dir}/mesh.pcd')
		# for i, dpt_file in enumerate(dpt_gt_files):
		# 	print(f'Scene: {dpt_file}')
		# 	depth_map = read_depth_map(dpt_file, clip_start, clip_end)
		#
		#
		# 	## depth inpaint
		# 	# mask = (depth_map>1.51).astype(np.uint8)
		# 	# src = depth_map.copy().astype(np.float32)
		# 	# dst = cv2.inpaint(src, mask, 6, cv2.INPAINT_NS)
		# 	# depth_map = dst.astype(np.float64)
		# 	# depth_map[depth_map > 1.51] = 1.5
		# 	# depth_gt_map = read_depth_map(dpt_gt_files[i], clip_start, clip_end)
		# 	# depth_map[depth_map>outlier_dict[obj_name]] = depth_gt_map[depth_map>outlier_dict[obj_name]]
		#
		# 	depth_map[depth_map<0.1] = 0
		# 	depth_map[depth_map>outlier_dict[obj_name]] = 0
		#
		# 	# mask = (depth_map<0.1).astype(np.uint8)
		# 	# depth_map = cv2.inpaint(depth_map.astype(np.float32), mask, 3, cv2.INPAINT_NS)
		#
		# 	# valid_mask = depth_map >= 0.1
		# 	segment_map = cv2.imread(seg_files[i],-1)
		# 	with open(gt_files[i], 'r') as f:
		# 		meta = json.load(f)
		#
		# 	xyz_map = depth2xyzmap(depth_map, K)
		# 	if obj_name == 'gear':
		# 		xyz_map = xyz_map[::2, ::2]
		# 		segment_map = segment_map[::2, ::2]
		# 	# cloud_xyz_full = xyz_map[valid_mask].reshape(-1, 3)
		# 	# cloud_nml_full = fill_depth_normal(cloud_xyz_full)
		# 	# cloud_seg_full = segment_map[valid_mask].reshape(-1)
		#
		# 	input = {
		# 		'points': []
		# 	}
		# 	for j in range(len(meta)):
		# 		seg_id = meta[j]['segmentation_id']
		# 		occlusion_rate = meta[j]['occlusion_rate']
		# 		if occlusion_rate >= 0.5:
		# 			continue
		#
		# 		roi_mask = (segment_map == seg_id).astype('uint8')
		# 		# roi_mask_def = defor_2D(roi_mask, 1, 1)
		# 		# mask = roi_mask.copy()
		# 		# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
		# 		# mask_erode = cv2.erode(mask, kernel, iterations=3)  # rand_r
		# 		# roi_mask_def = mask_erode.copy()
		#
		# 		mask = (roi_mask != 0) & (xyz_map[:, :, 2] > 0.1)
		# 		cloud_xyz = xyz_map[mask]
		#
		# 		cloud_xyz_ins = pose_predictor.sample_points(cloud_xyz)
		# 		input['points'].append(cloud_xyz_ins)
		#
		# 	pred_lst = []
		# 	pred_pp_lst = []
		# 	if len(input['points']) != 0:
		# 		input['points'] = np.asarray(input['points'])
		# 		p_RT, p_Vote = pose_predictor.predict(input)
		#
		# 		for k in range(len(input['points'])):
		# 			R, t = p_RT[k][:3, :3], p_RT[k][:3, 3]
		# 			ob_in_cam = np.eye(4)
		# 			ob_in_cam[:3, :3] = p_RT[k][:3, :3]
		# 			ob_in_cam[:3, 3] = p_RT[k][:3, 3]
		# 			ob_in_world = cam_in_world @ ob_in_cam
		#
		# 			target = copy.deepcopy(mesh_pcd)
		# 			points_in_cam = copy.deepcopy(np.asarray(input['points'][k]))
		# 			source = toOpen3dCloud(points_in_cam)
		# 			trans_init = np.linalg.inv(ob_in_cam)
		# 			reg_p2p = o3d.pipelines.registration.registration_icp(
		# 				source, target, 0.005, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
		# 			ob_in_cam_pp = np.linalg.inv(reg_p2p.transformation)
		# 			ob_in_world_pp = cam_in_world @ ob_in_cam_pp
		#
		# 			pred_lst.append({
		# 				'R': ob_in_world[:3, :3].tolist(),
		# 				'score': 1.0,
		# 				't': ob_in_world[:3, 3].tolist()
		# 			})
		#
		# 			pred_pp_lst.append({
		# 				'R': ob_in_world_pp[:3, :3].tolist(),
		# 				'score': 1.0,
		# 				't': ob_in_world_pp[:3, 3].tolist()
		# 			})
		#
		# 	result_file = os.path.join(result_folder, os.path.basename(dpt_file).split('.')[0]+'.json')
		# 	with open(result_file, 'w') as f:
		# 		json.dump(pred_lst, f)
		#
		# 	result_file = os.path.join(result_pp_folder, os.path.basename(dpt_file).split('.')[0] + '.json')
		# 	with open(result_file, 'w') as f:
		# 		json.dump(pred_pp_lst, f)

