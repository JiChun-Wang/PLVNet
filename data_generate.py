"""
Data Generation:
Input: Object model (.ply); Camera Configuration (K, H, W, pose_in_world); The total number (The least and most number)
       Save dir
"""
import os
code_dir = os.path.dirname(os.path.realpath(__file__))
import argparse
import shutil
import glob

import numpy as np
import open3d as o3d

from Data_Synthesis.env import Env, Env_v2
import Data_Synthesis.utils as PU


def parse_camera_config(file):
	with open(file) as f:
		lines = f.readlines()

	cam_params = {}
	for i in range(len(lines)):
		line = lines[i].split()
		if line[0] == 'location':
			value = [float(line[1]), float(line[2]), float(line[3])]
		elif line[0] == 'rotation':
			value = [float(line[1]), float(line[2]), float(line[3]), float(line[4])]
		else:
			if line[0] in ['width', 'height']:
				value = int(line[1])
			else:
				value = float(line[1])
		cam_params[line[0]] = value
	return cam_params


def ply2obj(ply_pth):
	mesh = o3d.io.read_triangle_mesh(ply_pth)
	obj_pth = ply_pth.replace('.ply', '.obj')
	o3d.io.write_triangle_mesh(obj_pth, mesh)
	return obj_pth


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='Sileance_Dataset', help='indicate dataset name')
	parser.add_argument('--model_name', type=str, default='gear', help='indicate object name')
	parser.add_argument('--min_inst_num', type=int, default=30, help='indicate the min instance num')
	parser.add_argument('--max_inst_num', type=int, default=50
	                    , help='indicate the max instance num')
	parser.add_argument('--num_cycles_train', type=int, default=100)
	parser.add_argument('--num_cycles_val', type=int, default=10)
	args = parser.parse_args()

	dataset, model_name = args.dataset, args.model_name
	min_instance, max_instance = args.min_inst_num, args.max_inst_num
	num_cycles_train, num_cycles_val = args.num_cycles_train, args.num_cycles_val

	data_dir = f'{code_dir}/data/{dataset}/{model_name}'
	assert os.path.exists(data_dir), 'The directory of given model does not exist!'

	obj_model_pth = os.path.join(data_dir, 'mesh.obj')
	if not os.path.exists(obj_model_pth):
		ply_model_pth = os.path.join(data_dir, 'mesh.ply')
		obj_model_pth = ply2obj(ply_model_pth)
	bin_model_pth = os.path.join(code_dir, f'data/{dataset}/box.obj')
	cam_model_pth = os.path.join(code_dir, f'data/{dataset}/kinect_sensor.obj')
	cam_para_pth = os.path.join(data_dir, 'camera_params.txt')

	cam_params = parse_camera_config(cam_para_pth)

	n_train = 10000
	n_val = 1
	for split in ['val_vis']:
		curr_data_dir = f'{data_dir}/{split}'
		if not os.path.exists(curr_data_dir):
			os.makedirs(curr_data_dir, exist_ok=True)
		start_idx = len(glob.glob(f'{curr_data_dir}/*meta.pkl'))

		# if split == 'train':
		# 	ids = np.arange(start_idx, n_train)
		# else:
		# 	ids = np.arange(start_idx, n_val)

		if os.path.exists(curr_data_dir):
			os.system(f"rm -rf {curr_data_dir}")
		os.makedirs(curr_data_dir)
		if split == 'train':
			ids = np.arange(n_train)
		else:
			ids = np.arange(n_val)

		env = Env(bin_model_pth, cam_model_pth, cam_params, gui=True)
		for i, scene_id in enumerate(ids):
			num_instances = np.random.randint(min_instance, max_instance)
			env.generate_one(obj_model_pth, num_instances, scene_id, curr_data_dir)
			env.reset()
			# if i == 300:
			# 	break
			if (i!=0) and (i%25==0):
				del env
				env = Env(bin_model_pth, cam_model_pth, cam_params, gui=False)


	# for split in ['train', 'val']:
	# 	split_data_dir = f'{data_dir}/{split}'
	# 	if os.path.exists(split_data_dir):
	# 		os.system(f"rm -rf {split_data_dir}")
	# 	os.makedirs(split_data_dir)
	# 	if split == 'train':
	# 		num_cycles = num_cycles_train
	# 	else:
	# 		num_cycles = num_cycles_val
	#
	# 	env = Env(bin_model_pth, cam_model_pth, cam_params, gui=False)
	# 	for idx_cycle in range(1, num_cycles+1):
	# 		curr_data_dir = f'{split_data_dir}/{idx_cycle:04d}'
	# 		os.makedirs(curr_data_dir)
	#
	# 		for num_instance in range(min_instance, max_instance+1):
	# 			env.generate_one(obj_model_pth, num_instance, num_instance, curr_data_dir)
	# 			env.reset()
	# 			if num_instance % 20 == 0 :
	# 				del env
	# 				env = Env(bin_model_pth, cam_model_pth, cam_params, gui=False)

	# for split in ['train', 'val']:
	# 	split_data_dir = f'{data_dir}/{split}'
	# 	if os.path.exists(split_data_dir):
	# 		os.system(f"rm -rf {split_data_dir}")
	# 	os.makedirs(split_data_dir)
	# 	if split == 'train':
	# 		num_cycles = num_cycles_train
	# 	else:
	# 		num_cycles = num_cycles_val
	#
	# 	env = Env_v2(bin_model_pth, cam_model_pth, cam_params, gui=True)
	# 	for idx_cycle in range(43, 44):
	# 		curr_data_dir = f'{split_data_dir}/{idx_cycle:04d}'
	# 		os.makedirs(curr_data_dir)
	#
	# 		for num_instance in range(min_instance, max_instance+1):
	# 			env.generate_one(obj_model_pth, num_instance, num_instance, curr_data_dir)
	# 			env.reset()
	# 			if num_instance % 20 == 0 :
	# 				del env
	# 				env = Env_v2(bin_model_pth, cam_model_pth, cam_params, gui=True)
	# 	break