import copy
import os
code_dir = os.path.dirname(os.path.realpath(__file__))
import argparse
import numpy as np
import os, glob, cv2, trimesh, pickle, gzip, re


import open3d as o3d
from PIL import Image
from transformations import *
from scipy.spatial import cKDTree

from util.offscreen_renderer import ModelRendererOffscreen
from util.utils import toOpen3dCloud, depth2xyzmap, correct_pcd_normal_direction, read_normal_image


def fill_depth_normal_worker(depth_file):
	depth = cv2.imread(depth_file, -1) / 1e3
	depth[depth<0.1] = 0
	depth[depth>3.] = 0
	with open(depth_file.replace('depth.png', 'meta.pkl'), 'rb') as ff:
		meta = pickle.load(ff)
		K = meta['K']

	xyz_map = depth2xyzmap(depth, K)
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
	out_file = depth_file.replace('depth', 'normal')
	Image.fromarray(normal_map).save(out_file)
	print(f"Write to {out_file}")


def fill_depth_normal():
	for split in ['train', 'val']:
		depth_files = sorted(glob.glob(f'{code_dir}/data/{dataset}/{obj_name}/{split}/*depth.png'))
		print("#depth_files={}".format(len(depth_files)))

		for depth_file in depth_files:
			fill_depth_normal_worker(depth_file)


def compute_per_ob_visibility_worker(color_file):
	with open(color_file.replace('rgb.png', 'meta.pkl'), 'rb') as ff:
		meta = pickle.load(ff)
	seg = cv2.imread(color_file.replace('rgb', 'seg'), -1)
	seg_ids = np.unique(seg)

	K = np.array(meta['K']).reshape(3, 3)
	renderer = ModelRendererOffscreen([model_dir], K, H=meta['H'], W=meta['W'])
	cam_in_world = meta['cam_in_world']

	visual_ratio_dict = {}
	for seg_id in seg_ids:
		if seg_id in meta['env_body_ids']:
			continue
		ob_in_world = meta['poses'][seg_id]
		ob_in_cam = np.linalg.inv(cam_in_world) @ ob_in_world
		color, depth = renderer.render([ob_in_cam])
		visual_ratio = float(np.sum(seg==seg_id)) / (float(np.sum(depth>=0.1))+1e-6)
		visual_ratio_dict[seg_id] = max(min(visual_ratio, 1), 0)

	meta['visibility_rate'] = visual_ratio_dict
	meta_path = color_file.replace('rgb.png', 'meta.pkl')
	print(f'Write to {meta_path}')
	with open(meta_path, 'wb') as ff:
		pickle.dump(meta, ff)


def compute_per_ob_visibility():
	for split in ['train', 'val']:
		color_files = sorted(glob.glob(f'{code_dir}/data/{dataset}/{obj_name}/{split}/*rgb.png'))
		print('#color_files={}'.format(len(color_files)))

		for i, color_file in enumerate(color_files):
			compute_per_ob_visibility_worker(color_file)


def make_crop_scene_dataset_worker(color_file, out_dir, downsample_size):
	print(color_file)
	index_str = re.findall(r'[0-9]{5}', color_file)[0]

	rgb = np.array(Image.open(color_file))
	depth = cv2.imread(color_file.replace('rgb', 'depth'), -1) / 1e3
	depth[depth<0.1] = 0
	depth[depth>3] = 0
	seg_map = cv2.imread(color_file.replace('rgb', 'seg'), -1).astype(int)
	with open(color_file.replace('rgb.png', 'meta.pkl'), 'rb') as ff:
		meta = pickle.load(ff)
		K = meta['K']
		env_body_ids = meta['env_body_ids']

	valid_mask = depth >= 0.1
	cloud_rgb_origin = rgb[valid_mask].reshape(-1,3)
	xyz_map = depth2xyzmap(depth, K)
	cloud_xyz_origin = xyz_map[valid_mask].reshape(-1, 3)
	normal_map = read_normal_image(color_file.replace('rgb', 'normal'))
	cloud_nml_origin = normal_map[valid_mask].reshape(-1, 3)
	cloud_seg_origin = seg_map[valid_mask].reshape(-1)

	pcd = toOpen3dCloud(cloud_xyz_origin)
	downpcd = pcd.voxel_down_sample(voxel_size=downsample_size)
	pts = np.asarray(downpcd.points).copy()
	kdtree = cKDTree(cloud_xyz_origin)
	dists, indices = kdtree.query(pts)

	cloud_xyz = cloud_xyz_origin[indices]
	cloud_rgb = cloud_rgb_origin[indices]
	cloud_nml = cloud_nml_origin[indices]
	cloud_seg = cloud_seg_origin[indices]

	# num_pts = len(cloud_xyz)
	# indices = np.random.choice(num_pts, 250000, replace=num_pts<250000)
	# cloud_xyz = cloud_xyz[indices]
	# cloud_rgb = cloud_rgb[indices]
	# cloud_nml = cloud_nml[indices]
	# cloud_seg = cloud_seg[indices]

	coords = np.ascontiguousarray(cloud_xyz-cloud_xyz.mean(0))
	colors = np.ascontiguousarray(cloud_nml)

	seg_ids = np.unique(cloud_seg)
	sem_labels = np.zeros(coords.shape[0])
	instance_labels = np.ones(coords.shape[0]) * -100
	num_instance = 0
	for seg_id in seg_ids:
		if seg_id in env_body_ids:
			continue
		if meta['visibility_rate'][seg_id] >= 0.5:
			sem_labels[cloud_seg==seg_id] = 1
			instance_labels[cloud_seg==seg_id] = num_instance
			num_instance += 1

	infos = {
		'points': coords,
		'normals': colors,
		'semantic_labels': sem_labels,
		'instance_labels': instance_labels
	}

	with gzip.open(f'{out_dir}/{index_str}.pkl', 'wb') as ff:
		pickle.dump(infos, ff)

	if int(index_str) < 5:
		# pcd = toOpen3dCloud(cloud_xyz, cloud_rgb, cloud_nml)
		# o3d.io.write_point_cloud(f'{out_dir}/{index_str}.ply', pcd)

		#### visualization for semantic labels ###########
		bg = np.array([1., 0., 0.])
		fg = 1 - bg
		rgb = []
		for i in sem_labels:
			if i == 0.:
				rgb.append(bg)
			else:
				rgb.append(fg)
		rgb = np.array(rgb)
		pcd = toOpen3dCloud(cloud_xyz, rgb, cloud_nml)
		o3d.io.write_point_cloud(f'{out_dir}/{index_str}.ply', pcd)

		#### visualization for instance labels ###########
		# inst_ids = np.unique(instance_labels)
		# rgb = np.zeros((len(instance_labels), 3))
		# for inst_id in inst_ids:
		# 	rgb[instance_labels==inst_id, :] = np.random.rand(3)
		# pcd = toOpen3dCloud(cloud_xyz, rgb, cloud_nml)
		# o3d.io.write_point_cloud(f'{out_dir}/{index_str}.ply', pcd)


def make_crop_scene_dataset(downsample_size):
	'''For instance segmentation training, remove background e.g. bin
	'''
	for split in ['train', 'val']:
		color_files = sorted(glob.glob(f'{code_dir}/data/{dataset}/{obj_name}//{split}/*rgb.png'))
		print(f'color_files={len(color_files)}')

		out_dir = f'{code_dir}/data/{dataset}/{obj_name}/{split}_instance_segmentation'
		print(f'out_dir: {out_dir}')
		os.system(f'rm -rf {out_dir} && mkdir -p {out_dir}')

		for i, color_file in enumerate(color_files):
			make_crop_scene_dataset_worker(color_file, out_dir, downsample_size)
		# 	if i == 5:
		# 		break
		# break


def make_isolated_training_data_worker(depth_file, out_dir, downsample_size=0.001):
	'''
	Isolate objects in the scene
	'''
	print('depth_file', depth_file)
	index_str = re.findall(r'[0-9]{5}', depth_file)[0]

	depth = cv2.imread(depth_file, -1) / 1e3
	depth[depth < 0.1] = 0
	depth[depth > 3] = 0
	normal_map = read_normal_image(depth_file.replace('depth', 'normal'))
	seg_map = cv2.imread(depth_file.replace('depth', 'seg'), -1).astype(int)
	with open(depth_file.replace('depth.png', 'meta.pkl'), 'rb') as ff:
		meta = pickle.load(ff)
		K = meta['K']
		env_body_ids = meta['env_body_ids']

	xyz_map = depth2xyzmap(depth, K)

	seg_ids = np.unique(seg_map)
	for seg_id in seg_ids:
		if seg_id in env_body_ids:
			continue
		if meta['visibility_rate'][seg_id] < 0.5:
			continue

		mask = (seg_map == seg_id) & (depth >= 0.1)
		cloud_xyz = xyz_map[mask].reshape(-1, 3)
		cloud_nml = normal_map[mask].reshape(-1, 3)

		pcd = toOpen3dCloud(cloud_xyz)
		downpcd = pcd.voxel_down_sample(voxel_size=downsample_size)
		pts = np.asarray(downpcd.points).copy()
		kdtree = cKDTree(cloud_xyz)
		dists, indices = kdtree.query(pts)

		cloud_xyz = cloud_xyz[indices]
		cloud_nml = cloud_nml[indices]

		ob_in_world = meta['poses'][seg_id]
		cam_in_world = meta['cam_in_world']
		ob_in_cam = np.linalg.inv(cam_in_world)@ob_in_world
		rotation, translation = ob_in_cam[:3, :3], ob_in_cam[:3, 3]

		out_data = {'cloud_xyz': cloud_xyz, 'cloud_normal': cloud_nml, 'rotation': rotation, 'translation': translation,
		           'depth_file': depth_file, 'seg_id': seg_id}

		out_path = f'{out_dir}/{index_str}_seg{seg_id}.pkl'
		with gzip.open(out_path, 'wb') as ff:
			pickle.dump(out_data, ff)


def make_isolated_training_data():
	for split in ['train', 'val']:
		depth_files = sorted(glob.glob(f'{code_dir}/data/{dataset}/{obj_name}/{split}/*depth.png'))
		print('There are {} depth_files in {} split! '.format(len(depth_files), split))

		out_dir = f'{code_dir}/data/{dataset}/{obj_name}/{split}_pose_estimation'
		print(f'out_dir: {out_dir}')
		os.system(f'rm -rf {out_dir} && mkdir -p {out_dir}')

		for i, depth_file in enumerate(depth_files):
			make_isolated_training_data_worker(depth_file, out_dir)
			if i == 5000:
				break
		# 	if i == 5:
		# 		break
		# break


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='Sileance_Dataset', help='indicate dataset name')
	parser.add_argument('--obj_name', type=str, default='gear', help='indicate object name')
	args = parser.parse_args()

	dataset = args.dataset
	obj_name = args.obj_name

	model_dir = os.path.join(code_dir, f'data/{dataset}/{obj_name}/mesh.obj')

	# fill_depth_normal()
	compute_per_ob_visibility()
	make_crop_scene_dataset(downsample_size=0.005)
	# make_isolated_training_data()
