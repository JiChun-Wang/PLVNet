import glob
import os, time

import cv2

code_dir = os.path.dirname(os.path.realpath(__file__))
import argparse

import numpy as np
import torch
import spconv
from scipy.spatial import cKDTree
from sklearn.cluster import MeanShift
import open3d as o3d

# from PointGroup.util.config import cfg
from PointGroup.lib.pointgroup_ops.functions import pointgroup_ops
from PointGroup.model.pointgroup.pointgroup import PointGroup
from util.utils import toOpen3dCloud


class PointGroupPredictor:
	def __init__(self, cfg):
		self.full_scale = cfg.full_scale
		self.scale = cfg.scale
		self.mode = cfg.mode
		self.cfg = cfg

		self.model = PointGroup(cfg)
		total_params = sum(p.numel() for p in self.model.parameters())
		print("Total parameters:", total_params)
		checkpoint = torch.load(f'{cfg.exp_dir}/pointgroup_best_train.pth.tar')
		print('Instance Segmentation: loading model checkpoint from epoch {}'.format(checkpoint['epoch']))
		self.model.load_state_dict(checkpoint['state_dict'])
		self.model.cuda().eval()

	def predict(self, data, idx=0, bandwidth=None, vis_filter=True):
		cloud_xyz = data['cloud_xyz']
		# cloud_nml = data['cloud_nml']

		with torch.no_grad():
			locs = []
			locs_float = []
			feats = []

			batch_offsets = [0]
			for i in [0]:
				xyz_origin = np.ascontiguousarray(cloud_xyz - cloud_xyz.mean(0))
				# normals = np.ascontiguousarray(cloud_nml)

				xyz = xyz_origin * self.scale
				xyz -= xyz.min(0)

				batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

				locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
				locs_float.append(torch.from_numpy(xyz_origin))
				# feats.append(torch.from_numpy(normals))
				feats.append(torch.from_numpy(xyz_origin))

			batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)

			locs = torch.cat(locs, 0)
			locs_float = torch.cat(locs_float, 0).to(torch.float32)
			feats = torch.cat(feats, 0)

			spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)

			voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, len(batch_offsets)-1, self.mode)


			coords = locs.cuda()  # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
			voxel_coords = voxel_locs.cuda()  # (M, 1 + 3), long, cuda
			p2v_map = p2v_map.cuda()  # (N), int, cuda
			v2p_map = v2p_map.cuda()  # (M, 1 + maxActive), int, cuda

			coords_float = locs_float.cuda()  # (N, 3), float32, cuda
			feats = feats.cuda()  # (N, C), float32, cuda

			if self.cfg.use_coords:
				feats = torch.cat((feats, coords_float), 1)
			voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, self.mode)  # (M, C), float, cuda

			input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, len(batch_offsets)-1)

			ret = self.model(input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, self.cfg.prepare_epochs)

			semantic_scores = ret['semantic_scores']  # (N, nClass=20) float32, cuda
			semantic_pred = semantic_scores.max(1)[1]  # (N) long, cuda
			sem_labels = semantic_pred.cpu().data.numpy()
			if self.cfg.save_semantic:
				os.makedirs(os.path.join(self.cfg.exp_dir, 'semantic'), exist_ok=True)
				out_dir = os.path.join(self.cfg.exp_dir, 'semantic')
				#### visualization for semantic labels ###########
				bg = np.array([1., 0., 0.])
				fg = np.array([0., 0., 1.])
				rgb = []
				for i in sem_labels:
					if i == 0.:
						rgb.append(bg)
					else:
						rgb.append(fg)
				rgb = np.array(rgb)
				# pcd = toOpen3dCloud(cloud_xyz, rgb, cloud_nml)
				pcd = toOpen3dCloud(cloud_xyz, rgb)
				o3d.io.write_point_cloud(f'{out_dir}/{idx}.ply', pcd)

			offsets = ret['pt_offsets'].cpu().data.numpy()
			xyz_original_all = coords_float.cpu().data.numpy()

			# if vis_filter:
			# 	xyz_fg_all = xyz_original_all[sem_labels==1]
			# 	offsets = offsets[sem_labels==1]
			# else:
			# 	xyz_fg_all = xyz_original_all.copy()
			# 	sem_labels = np.ones_like(sem_labels)
			xyz_fg_all = xyz_original_all[sem_labels==1]
			offsets = offsets[sem_labels==1]

			if len(xyz_fg_all) > 100:
				pcd = toOpen3dCloud(xyz_fg_all)
				pcd = pcd.voxel_down_sample(voxel_size=0.002 if self.cfg.obj_name == 'brick' else 0.01)
				xyz_down = np.asarray(pcd.points).copy()
				kdtree = cKDTree(xyz_fg_all)
				dists, indices = kdtree.query(xyz_down)
				xyz_down = xyz_fg_all[indices]
				xyz_shifted = xyz_down + offsets[indices]
				self.xyz_shifted = xyz_shifted

				os.makedirs(os.path.join(self.cfg.exp_dir, 'cluster'), exist_ok=True)
				out_dir = os.path.join(self.cfg.exp_dir, 'cluster')
				xyz_cluster = np.concatenate([xyz_down,xyz_shifted], axis=0)
				rgb_cluster = np.ones((len(xyz_cluster), 3))
				rgb_cluster[:len(xyz_down), :] = 0.9
				rgb_cluster[len(xyz_down):, :] = 0.1
				pcd_cluster = toOpen3dCloud(xyz_cluster, colors=rgb_cluster)
				o3d.io.write_point_cloud(f'{out_dir}/{idx}.ply', pcd_cluster)

				bw = bandwidth if bandwidth is not None else self.cfg.bandwidth
				# labels = MeanShift(bandwidth=bw, cluster_all=True, n_jobs=-1, seeds=None).fit_predict(xyz_shifted)
				labels = MeanShift(bandwidth=bw, bin_seeding=True, cluster_all=False, min_bin_freq=1).fit_predict(xyz_shifted)

				kdtree = cKDTree(xyz_down)
				dists, indices = kdtree.query(xyz_fg_all)
				ins_labels = labels[indices]

				if self.cfg.save_instance:
					os.makedirs(os.path.join(self.cfg.exp_dir, 'instance'), exist_ok=True)
					out_dir = os.path.join(self.cfg.exp_dir, 'instance')
					#### visualization for instance labels ###########
					inst_ids = np.unique(ins_labels)
					rgb = np.zeros((len(ins_labels), 3))
					for inst_id in inst_ids:
						rgb[ins_labels==inst_id, :] = np.random.rand(3)
					pcd = toOpen3dCloud(xyz_fg_all, rgb)
					o3d.io.write_point_cloud(f'{out_dir}/{idx}.ply', pcd)

				return sem_labels, ins_labels
			else:
				print('There is no one target object')
				return np.zeros((0,)), np.zeros((0,))


if __name__ == '__main__':
	from util.utils import parse_camera_config, read_depth_map, depth2xyzmap, fill_depth_normal, cloud_sample

	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='IPA_Dataset', help='indicate dataset name')
	parser.add_argument('--obj_name', type=str, default='IPARingScrew', help='indicate object name')
	args = parser.parse_args()
	dataset = args.dataset
	obj_name = args.obj_name

	outlier_dict = {"bunny": 1.70,
	                "candlestick": 1.10,
	                "pepper": 1.60,
	                "brick": 1.03,
	                "gear": 2.08,
	                "tless_20": 1.10,
	                "tless_22": 1.10,
	                "tless_29": 1.10}

	from PointGroup.util.config import get_parser as get_seg_cfg
	cfg_file = f'{code_dir}/PointGroup/exp/{dataset}/{obj_name}/config_pointgroup.yaml'
	cfg = get_seg_cfg(cfg_file)

	# data_dir = f'{cfg.data_root}/{dataset}/{obj_name}'
	# assert os.path.exists(data_dir), 'The target object does not exist!'

	# cam_para_pth = os.path.join(data_dir, 'camera_params.txt')
	# cam_params = parse_camera_config(cam_para_pth)
	# K = np.array([[cam_params['fu'], 0., cam_params['cu']],
	#               [0., cam_params['fv'], cam_params['cv']],
	#               [0., 0., 1.]])
	# clip_start = cam_params['clip_start']
	# clip_end = cam_params['clip_end']

	seg_predictor = PointGroupPredictor(cfg)

	# dpt_files = sorted(glob.glob(f'{data_dir}/depth/*.PNG'))
	# dpt_back_file = dpt_files[0]
	# dpt_file = dpt_files[1]
	# depth_back = read_depth_map(dpt_back_file, clip_start, clip_end)
	# depth = read_depth_map(dpt_file, clip_start, clip_end)

	# fill hole
	# depth_back[depth_back>1.51] = 0
	# depth[depth>1.51] = 0
	# kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
	# depth_back = cv2.morphologyEx(depth_back, cv2.MORPH_CLOSE, kernal)
	# depth = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernal)
	# mask = np.abs(depth-depth_back)<=0.005
	# depth[mask] = 0

	# depth[depth < 0.1] = 0
	# depth[depth > outlier_dict[obj_name]] = 0
	#
	# valid_mask = depth >= 0.1
	# xyz_map = depth2xyzmap(depth, K)
	# cloud_xyz_original = xyz_map[valid_mask].reshape(-1, 3)
	# cloud_nml_original = fill_depth_normal(cloud_xyz_original)

	# cloud_xyz_001, cloud_nml_001 = cloud_sample(cloud_xyz_original, cloud_nml_original, downsample_size=0.001)
	pcd = o3d.io.read_point_cloud(f"{code_dir}/real_world_scene.ply")
	cloud_xyz_original = np.float32(pcd.points)
	print(len(cloud_xyz_original))
	cloud_xyz_005, _ = cloud_sample(cloud_xyz_original, downsample_size=cfg.ds_size)
	# input = {'cloud_xyz' : cloud_xyz_005, 'cloud_nml' : cloud_nml_005}
	input = {'cloud_xyz': cloud_xyz_005}

	_ = seg_predictor.predict(input, idx=0)

	start_time = time.time()
	ins_label = seg_predictor.predict(input, idx=0)
	duration = time.time() - start_time
	print(f'Inference time: {duration}')
