import copy
import os, json
code_dir = os.path.dirname(os.path.realpath(__file__))
import numpy as np
import os, glob, cv2, trimesh, pickle, gzip, re

import open3d as o3d
from PIL import Image
from transformations import *
from scipy.spatial import cKDTree

from util.utils import *
from util.DataReader import DataReader
from util.offscreen_renderer import ModelRendererOffscreen


def fill_depth_normal_worker(depth_file, seg_file, datareader):
	depth = datareader.read_depth_map(depth_file)
	xyz_map = datareader.depth2xyzmap(depth)
	valid_mask = xyz_map[:, :, 2] >= 0.1
	pts = xyz_map[valid_mask].reshape(-1, 3)
	pcd = toOpen3dCloud(pts)
	pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.003, max_nn=30))
	pcd = correct_pcd_normal_direction(pcd)
	normals = np.asarray(pcd.normals).copy()
	normal_map = np.zeros(xyz_map.shape)
	vs, us = np.where(valid_mask > 0)
	normal_map[vs, us] = normals
	normal_map = np.round((normal_map + 1) / 2.0 * 255)
	normal_map = np.clip(normal_map, 0, 255).astype(np.uint8)
	return normal_map


def fill_depth_normal(data_dir, start_cycle=0, end_cycle=110):
	cam_para_path = os.path.join(data_dir, 'parameter.json')
	datareader = DataReader(cam_para_path)

	normal_folder = f'{data_dir}/p_normal'
	if not os.path.exists(normal_folder):
		os.makedirs(normal_folder)

	for idx_cycle in range(start_cycle, end_cycle):
		print(f'************** Cycle {idx_cycle} Start *****************')
		cycle_normal_folder = f'{normal_folder}/cycle_{idx_cycle:04d}'
		if not os.path.exists(cycle_normal_folder):
			os.makedirs(cycle_normal_folder)

		depth_files = sorted(glob.glob(f'{data_dir}/p_depth/cycle_{idx_cycle:04d}/*depth_uint16.png'))
		for depth_file in depth_files:
			normal_map = fill_depth_normal_worker(depth_file, datareader)
			obj_num = int(os.path.basename(depth_file).split('_')[0])
			out_file = f'{cycle_normal_folder}/{obj_num:03d}_normal.png'
			Image.fromarray(normal_map).save(out_file)
			print(f"Write to {out_file}")
		print(f'************** Cycle {idx_cycle} End *****************')


def compute_nunocs_label_worker(depth_file, seg_file, meta_file):
	dpt_map = cv2.imread(depth_file, -1) / 1e3
	seg_map = cv2.imread(seg_file, -1).astype(int)
	with open(meta_file, 'rb') as ff:
		meta = pickle.load(ff)

	H, W = dpt_map.shape
	K = meta['K']
	poses = meta['poses']

	model_name = meta['model']
	if model_name in ['hnm_ENG_CVM_CVM_T2032162201-000B']:
		scale = 0.75 * meta['scale']
	else:
		scale = meta['scale']
	model_file = f'{model_dir}/{model_name}/{model_name}.obj'
	model = trimesh.load(model_file)
	mesh_pts = model.vertices.copy() * scale
	max_xyz = mesh_pts.max(axis=0).reshape(1, 3)
	min_xyz = mesh_pts.min(axis=0).reshape(1, 3)
	center_xyz = (max_xyz + min_xyz) / 2

	diameter = np.sqrt(np.linalg.norm(max_xyz - min_xyz))

	xyz_map = depth2xyzmap(dpt_map, K)

	seg_ids = np.unique(seg_map)
	nocs_image = np.zeros((H, W, 3), dtype=np.uint8)
	nunocs_image = np.zeros((H, W, 3), dtype=np.uint8)
	for seg_id in seg_ids:
		if seg_id in [0, 1]:
			continue
		# if meta['visibility_rate'][seg_id] < 0.75:
		# 	continue

		valid_mask = (seg_map == seg_id) & (xyz_map[..., 2] >= 0.1)
		tmp_xyz = xyz_map[valid_mask].reshape(-1, 3)
		ob_in_world = poses[seg_id].copy()
		cam_in_world = meta['cam_in_world'].copy()
		ob_in_cam = np.linalg.inv(cam_in_world) @ ob_in_world
		cam_in_ob = np.linalg.inv(ob_in_cam)
		tmp_xyz = (cam_in_ob @ to_homo(tmp_xyz).T).T[:, :3]

		nocs_xyz = (tmp_xyz - center_xyz) / np.array([diameter, diameter, diameter]).reshape(1, 3)  # [-0.5,0.5]
		nocs_xyz = np.clip(nocs_xyz, -0.5, 0.5)
		nocs_image[valid_mask] = (nocs_xyz + 0.5) * 255

		nunocs_scale = 1.
		nunocs_xyz = (tmp_xyz - center_xyz) / (max_xyz - min_xyz).reshape(1, 3)  # [-0.5,0.5]
		nunocs_xyz = np.clip(nunocs_xyz, -0.5, 0.5)
		nunocs_xyz /= nunocs_scale
		nunocs_image[valid_mask] = (nunocs_xyz + 0.5) * 255

	return nocs_image, nunocs_image


def compute_nunocs_label():
	nunocs_folder = f'{source_dir}/p_nunocs'
	if not os.path.exists(nunocs_folder):
		os.makedirs(nunocs_folder)

	for idx_cycle in range(1, 501):
		print(f'************** Cycle {idx_cycle} Start *****************')
		cycle_nunocs_folder = f'{source_dir}/p_nunocs/cycle_{idx_cycle:04d}'
		if not os.path.exists(cycle_nunocs_folder):
			os.makedirs(cycle_nunocs_folder)

		depth_files = sorted(glob.glob(f'{source_dir}/p_depth/cycle_{idx_cycle:04d}/*depth.png'))
		segment_files = sorted(glob.glob(f'{source_dir}/p_segmentation/cycle_{idx_cycle:04d}/*segmentation.png'))
		gt_files = sorted(glob.glob(f'{source_dir}/gt/cycle_{idx_cycle:04d}/*.pkl'))
		assert len(depth_files) == len(
			segment_files), 'The numbers of depth files and segmentation files do not equal !'
		assert len(depth_files) == len(gt_files), 'The numbers of depth files and gt files do not equal !'
		for i in range(len(depth_files)):
			nocs_map, nunocs_map = compute_nunocs_label_worker(depth_files[i], segment_files[i], gt_files[i])
			obj_num = int(os.path.basename(depth_files[i]).split('_')[0])
			nocs_out_file = f'{cycle_nunocs_folder}/{obj_num:03d}_nocs.png'
			nunocs_out_file = f'{cycle_nunocs_folder}/{obj_num:03d}_nunocs.png'
			Image.fromarray(nocs_map).save(nocs_out_file)
			Image.fromarray(nunocs_map).save(nunocs_out_file)
			print(f"Write to {nunocs_out_file}")
		print(f'************** Cycle {idx_cycle} End *****************')
		# break


def compute_per_ob_visibility_worker(dpt_file, seg_file, gt_file):
	with open(gt_file, 'rb') as ff:
		meta = pickle.load(ff)
	seg = cv2.imread(seg_file, -1)
	seg_ids = np.unique(seg)

	K = np.array(meta['K']).reshape(3, 3)
	model_name = meta['model']
	model_file = f'{model_dir}/{model_name}/{model_name}.obj'
	if model_name in ['hnm_ENG_CVM_CVM_T2032162201-000B']:
		scale = 0.75 * meta['scale']
	else:
		scale = meta['scale']
	renderer = ModelRendererOffscreen([model_file], K, H=meta['H'], W=meta['W'], scale=scale)

	cam_in_world = meta['cam_in_world']
	visual_ratio_dict = {}
	for seg_id in seg_ids:
		if seg_id in [0, 1]:
			continue
		ob_in_world = meta['poses'][seg_id]
		ob_in_cam = np.linalg.inv(cam_in_world) @ ob_in_world
		ob_in_cam[0, 3] = 0
		ob_in_cam[1, 3] = 0
		color, depth = renderer.render([ob_in_cam])
		visual_ratio = float(np.sum(seg==seg_id)) / (float(np.sum(depth>=0.1))+1e-6)
		visual_ratio_dict[seg_id] = max(min(visual_ratio, 1), 0)

	meta['visibility_rate'] = visual_ratio_dict
	print(f'Write to {gt_file}')
	with open(gt_file, 'wb') as ff:
		pickle.dump(meta, ff)


def compute_per_ob_visibility():
	for idx_cycle in range(41, 61):
		print(f'************** Cycle {idx_cycle} Start *****************')
		depth_files = sorted(glob.glob(f'{source_dir}/p_depth/cycle_{idx_cycle:04d}/*depth.png'))
		segment_files = sorted(glob.glob(f'{source_dir}/p_segmentation/cycle_{idx_cycle:04d}/*segmentation.png'))
		gt_files = sorted(glob.glob(f'{source_dir}/gt/cycle_{idx_cycle:04d}/*.pkl'))
		assert len(depth_files) == len(
			segment_files), 'The numbers of depth files and segmentation files do not equal !'
		assert len(depth_files) == len(gt_files), 'The numbers of depth files and gt files do not equal !'
		for i in range(len(depth_files)):
			compute_per_ob_visibility_worker(depth_files[i], segment_files[i], gt_files[i])
		print(f'************** Cycle {idx_cycle} End *****************')


def make_crop_scene_dataset_worker(depth_file, seg_file, gt_file, idx, out_dir, downsample_size):
	depth_map = cv2.imread(depth_file, -1) / 1e3
	segment_map = cv2.imread(seg_file, -1).astype(int)
	with open(gt_file, 'rb') as ff:
		meta = pickle.load(ff)

	xyz_map = depth2xyzmap(depth_map, meta['K'])

	valid_mask = (segment_map != 0) & (segment_map != 1)
	if np.sum(valid_mask) == 0:
		return
	cloud_xyz_origin = xyz_map[valid_mask].reshape(-1, 3)
	cloud_seg_origin = segment_map[valid_mask].reshape(-1)

	pcd = toOpen3dCloud(cloud_xyz_origin)
	downpcd = pcd.voxel_down_sample(voxel_size=downsample_size)
	pts = np.asarray(downpcd.points).copy()
	kdtree = cKDTree(cloud_xyz_origin)
	dists, indices = kdtree.query(pts)

	cloud_xyz = cloud_xyz_origin[indices]
	cloud_seg = cloud_seg_origin[indices]

	coords = np.ascontiguousarray(cloud_xyz-cloud_xyz.mean(0))

	seg_ids = np.unique(cloud_seg)
	sem_labels = np.zeros(coords.shape[0])
	instance_labels = np.ones(coords.shape[0]) * -100
	num_instance = 0
	for seg_id in seg_ids:
		if seg_id in [0, 1]:
			continue
		if meta['visibility_rate'][seg_id] >= 0.75:
			sem_labels[cloud_seg==seg_id] = 1
			instance_labels[cloud_seg==seg_id] = num_instance
			num_instance += 1

	infos = {
		'points': coords,
		'semantic_labels': sem_labels,
		'instance_labels': instance_labels
	}

	print(f'Writing into {out_dir}/{idx:05d}.pkl')
	with gzip.open(f'{out_dir}/{idx:05d}.pkl', 'wb') as ff:
		pickle.dump(infos, ff)


def make_crop_scene_dataset(source_dir, downsample_size):
	'''For instance segmentation training, remove background e.g. bin
	'''
	for split in ['train', 'val']:
		out_dir = f'{code_dir}/data/{dataset}/{obj_name}/{split}_instance_segmentation'
		print(f'out_dir: {out_dir}')
		os.system(f'rm -rf {out_dir} && mkdir -p {out_dir}')

		if split == 'train':
			input_cycles = list(range(1, 125)) + list(range(126, 250)) + list(range(251, 375)) + list(range(376, 500))
		else:
			input_cycles = [125, 250, 375, 500]

		count = 0
		for idx_cycle in input_cycles:
			print(f'************** {split} Cycle {idx_cycle} Start *****************')
			depth_files = sorted(glob.glob(f'{source_dir}/p_depth/cycle_{idx_cycle:04d}/*depth.png'))
			segment_files = sorted(glob.glob(f'{source_dir}/p_segmentation/cycle_{idx_cycle:04d}/*segmentation.png'))
			gt_files = sorted(glob.glob(f'{source_dir}/gt/cycle_{idx_cycle:04d}/*.pkl'))
			assert len(depth_files) == len(segment_files), 'The numbers of depth files and segmentation files do not equal !'
			assert len(depth_files) == len(gt_files), 'The numbers of depth files and gt files do not equal !'
			for i in range(len(depth_files)):
				make_crop_scene_dataset_worker(depth_files[i], segment_files[i], gt_files[i], count, out_dir, downsample_size)
				count += 1
			print(f'************** {split} Cycle {idx_cycle} End *****************')
		# 	break
		# break


def get_fs_net_scale(c, scale):
	if c == 'hnm_ENG_CVM_CVM_T2032162201-000B':
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

	mean_shape = np.array([57.900, 27.065, 35.946]) * 1.75
	fsnet_scale = np.array([x, y, z]) * scale - mean_shape
	return fsnet_scale / 1000., mean_shape / 1000.


def make_isolated_training_data_worker(depth_file, nocs_file, nunocs_file, segment_file, gt_file, idx, out_dir):
	'''
	Isolate objects in the scene
	'''
	depth_map = cv2.imread(depth_file, -1) / 1e3
	nocs_map = np.array(Image.open(nocs_file))
	nunocs_map = np.array(Image.open(nunocs_file))
	segment_map = cv2.imread(segment_file, -1).astype(int)
	with open(gt_file, 'rb') as ff:
		meta = pickle.load(ff)

	xyz_map = depth2xyzmap(depth_map, meta['K'])

	seg_ids = np.unique(segment_map)
	for seg_id in seg_ids:
		if seg_id in [0, 1]:
			continue
		if meta['visibility_rate'][seg_id] < 0.75:
			continue

		roi_mask = (segment_map==seg_id).astype('uint8')
		mask = (roi_mask!=0) & (xyz_map[:,:,2]>=0.1)
		cloud_xyz = xyz_map[mask].reshape(-1, 3)
		cloud_nocs = nocs_map[mask].reshape(-1, 3) / 255.0
		cloud_nunocs = nunocs_map[mask].reshape(-1, 3) / 255.0

		ob_in_world = meta['poses'][seg_id]
		cam_in_world = meta['cam_in_world']
		ob_in_cam = np.linalg.inv(cam_in_world) @ ob_in_world
		rotation, translation = ob_in_cam[:3, :3], ob_in_cam[:3, 3]

		fsnet_scale, mean_shape = get_fs_net_scale(meta['model'], meta['scale'])

		out_data = {'cloud_xyz': cloud_xyz,
		            'cloud_nocs': cloud_nocs,
		            'cloud_nunocs': cloud_nunocs,
		            'rotation': rotation,
		            'translation': translation,
		            'fsnet_scale': fsnet_scale,
		            'mean_shape': mean_shape}

		out_path = f'{out_dir}/{idx:05d}_seg{seg_id}.pkl'
		print(f'Writing into {out_path}')
		with gzip.open(out_path, 'wb') as ff:
			pickle.dump(out_data, ff)


def make_isolated_training_data(source_dir):
	for split in ['train', 'val']:
		out_dir = f'{code_dir}/data/{dataset}/{obj_name}/{split}_pose_estimation'
		print(f'out_dir: {out_dir}')
		os.system(f'rm -rf {out_dir} && mkdir -p {out_dir}')

		cycles_per_object = 60
		if split == 'train':
			input_cycles = list(range(1, cycles_per_object)) + list(range(126, 125+cycles_per_object)) + \
									list(range(251, 250+cycles_per_object)) + list(range(376, 375+cycles_per_object))
		else :
			input_cycles = [125, 250, 375, 500]

		count = 0
		# count = len(glob.glob(f'{out_dir}/*.pkl'))
		# print(count)
		for idx_cycle in input_cycles :
			print(f'************** {split} Cycle {idx_cycle} Start *****************')
			depth_files = sorted(glob.glob(f'{source_dir}/p_depth/cycle_{idx_cycle:04d}/*_depth.png'))
			nocs_files = sorted(glob.glob(f'{source_dir}/p_nunocs/cycle_{idx_cycle:04d}/*_nocs.png'))
			nunocs_files = sorted(glob.glob(f'{source_dir}/p_nunocs/cycle_{idx_cycle:04d}/*_nunocs.png'))
			segment_files = sorted(glob.glob(f'{source_dir}/p_segmentation/cycle_{idx_cycle:04d}/*_segmentation.png'))
			gt_files = sorted(glob.glob(f'{source_dir}/gt/cycle_{idx_cycle:04d}/*.pkl'))
			assert len(depth_files) == len(
				segment_files), 'The numbers of depth files and segmentation files do not equal !'
			assert len(depth_files) == len(gt_files), 'The numbers of depth files and gt files do not equal !'
			for i in range(len(depth_files)):
				make_isolated_training_data_worker(depth_files[i], nocs_files[i], nunocs_files[i], segment_files[i], gt_files[i], count, out_dir)
				count += 1
				# if count == 60:
				# 	break
			print(f'************** {split} Cycle {idx_cycle} End *****************')
		# 	break
		# break


if __name__ == "__main__":
	# source_dir = '/media/ubuntu/My Passport/Datasets/Cat_Bin_Picking/training'
	# model_dir = '/media/ubuntu/My Passport/Datasets/Cat_Bin_Picking/model'
	source_dir = '/dataset/Bin_Picking/CatBP_Dataset/all'
	model_dir = '/dataset/Bin_Picking/CatBP_Dataset/all/model'
	dataset = 'CatBP_Dataset'
	obj_name = 'all'
	# fill_depth_normal(source_dir, start_cycle=0, end_cycle=350)
	compute_per_ob_visibility()
	# compute_nunocs_label()
	# make_crop_scene_dataset(source_dir, downsample_size=0.005)
	# make_isolated_training_data(source_dir)
