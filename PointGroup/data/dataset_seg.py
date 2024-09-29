import math, glob, gzip, pickle, random
import numpy as np

import torch
from torch.utils.data import DataLoader

# from ..util.config import cfg
from ..lib.pointgroup_ops.functions import pointgroup_ops


class Dataset(torch.utils.data.Dataset):
	def __init__(self, cfg, phase='train'):
		super().__init__()
		assert phase in ['train', 'val'], 'The dataset is only for training or validating! '
		self.phase = phase

		self.full_scale = cfg.full_scale
		self.scale = cfg.scale
		self.max_npoint = cfg.max_npoint
		self.mode = cfg.mode

		data_dir = cfg.data_root
		dataset = cfg.dataset
		obj_name = cfg.obj_name
		if phase == 'train':
			self.files = sorted(glob.glob(f"{data_dir}/{dataset}/{obj_name}/train_instance_segmentation/*.pkl"))
		else:
			self.files = sorted(glob.glob(f"{data_dir}/{dataset}/{obj_name}/val_instance_segmentation/*.pkl"))

		print("phase: {}, num files={}".format(phase,len(self.files)))

	def __len__(self):
		return len(self.files)

	def getInstanceInfo(self, xyz, instance_label):
		instance_info = np.ones((xyz.shape[0], 9), dtype=np.float32) * -100.0
		instance_pointnum = []
		instance_num = int(instance_label.max()) + 1
		for i_ in range(instance_num):
			inst_idx_i = np.where(instance_label == i_)[0]

			xyz_i = xyz[inst_idx_i]
			min_xyz_i = xyz_i.min(0)
			max_xyz_i = xyz_i.max(0)
			mean_xyz_i = xyz_i.mean(0)
			instance_info_i = instance_info[inst_idx_i]
			instance_info_i[:, 0:3] = mean_xyz_i
			instance_info_i[:, 3:6] = min_xyz_i
			instance_info_i[:, 6:9] = max_xyz_i
			instance_info[inst_idx_i] = instance_info_i

			instance_pointnum.append(inst_idx_i[0].size)

		return instance_num, {"instance_info": instance_info, "instance_pointnum": instance_pointnum}

	def __getitem__(self, index):
		while 1:
			with gzip.open(self.files[index], 'rb') as ff :
				data = pickle.load(ff)
			cloud_xyz = data['points']
			if len(cloud_xyz) != 0:
				break
			print(index)
			return self.__getitem__((index + 1) % self.__len__())
		return index

	# def dataAugment(self, xyz, jitter=False, flip=False, rot=False):
	# 	m = np.eye(3)
	# 	if jitter:
	# 		m += np.random.randn(3, 3) * 0.01
	# 	if flip:
	# 		m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
	# 	if rot :
	# 		# theta = np.random.rand() * 2 * math.pi
	# 		theta = random.choice([-math.pi / 2, 0, math.pi / 2, math.pi])
	# 		m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0],
	# 						  [0, 0, 1]])  # rotation
	# 	return np.matmul(xyz, m)

	def dataAugment(self, xyz, label, instance_label, max_drop_ratio=0.2):
		num_pts = len(xyz)

		## dropout
		dropout_ratio = np.random.uniform(0, max_drop_ratio)
		n_drop = int(dropout_ratio * num_pts)
		drop_ids = np.random.choice(num_pts, size=n_drop, replace=False)
		keep_ids = list(set(np.arange(num_pts)) - set(drop_ids))
		xyz = xyz[keep_ids]
		label = label[keep_ids]
		instance_label = instance_label[keep_ids]

		## gaussian noise
		std = np.random.uniform(0, 0.003)
		xyz += np.random.normal(0, std, size=xyz.shape)

		return xyz, label, instance_label

	def merge(self, id):
		locs = []
		locs_float = []
		feats = []
		labels = []
		instance_labels = []

		instance_infos = []
		instance_pointnum = []

		batch_offsets = [0]

		total_inst_num = 0
		for i, idx in enumerate(id):
			with gzip.open(self.files[idx],'rb') as ff:
				data = pickle.load(ff)
			xyz_origin = data['points']
			label = data['semantic_labels']
			instance_label = data['instance_labels']

			if self.phase == 'train':
				xyz_origin, label, instance_label = self.dataAugment(xyz_origin, label, instance_label)

			xyz = xyz_origin * self.scale
			xyz -= xyz.min(0)

			inst_num, inst_infos = self.getInstanceInfo(xyz_origin, instance_label.astype(np.int32))
			inst_info = inst_infos["instance_info"]
			inst_pointnum = inst_infos["instance_pointnum"]

			instance_label[np.where(instance_label != -100)] += total_inst_num
			total_inst_num += inst_num

			batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

			locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
			locs_float.append(torch.from_numpy(xyz_origin))
			feats.append(torch.from_numpy(xyz_origin))
			labels.append(torch.from_numpy(label))
			instance_labels.append(torch.from_numpy(instance_label))

			instance_infos.append(torch.from_numpy(inst_info))
			instance_pointnum.extend(inst_pointnum)

		batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)

		locs = torch.cat(locs, 0)
		locs_float = torch.cat(locs_float, 0).to(torch.float32)
		feats = torch.cat(feats, 0)
		labels = torch.cat(labels, 0).long()
		instance_labels = torch.cat(instance_labels, 0).long()

		instance_infos = torch.cat(instance_infos, 0).to(torch.float32)
		instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)

		spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)
		voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, len(batch_offsets)-1, self.mode)

		return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
				'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
				'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
				'id': id, 'offsets': batch_offsets, 'spatial_shape': spatial_shape}
