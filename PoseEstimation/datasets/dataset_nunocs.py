import copy
import glob, pickle, time, gzip

import torch
from torch.utils.data import Dataset

from .data_augmentation import *


class NunocsPoseDataset(Dataset):
	def __init__(self, cfg, phase='train'):
		super().__init__()
		self.cfg = cfg
		assert phase in ['train', 'val'], 'The dataset is only for training or validating! '
		self.phase = phase

		data_dir = cfg.data_root
		dataset = cfg.dataset
		self.obj_name = cfg.obj_name
		self.num_point = cfg.num_pts
		if phase == 'train':
			self.files = sorted(glob.glob(f"{data_dir}/{dataset}/all/train_pose_estimation/*.pkl"))[
						 :cfg.num_train]
		else:
			self.files = sorted(glob.glob(f"{data_dir}/{dataset}/all/val_pose_estimation/*.pkl"))[:cfg.num_val]

		print("phase: {}, num files={}".format(phase, len(self.files)))

	def __len__(self):
		return len(self.files)

	def transform(self, data):
		cloud_xyz = data['cloud_xyz']
		if self.obj_name == 'nocs':
			cloud_nocs = data['cloud_nocs']
		elif self.obj_name == 'nunocs':
			cloud_nocs = data['cloud_nunocs']
		else:
			raise NotImplementedError

		replace = len(cloud_xyz) < self.num_point
		ids = np.random.choice(np.arange(len(cloud_xyz)), size=self.cfg.num_pts, replace=replace)
		cloud_xyz = cloud_xyz[ids].reshape(-1, 3)
		cloud_nocs = cloud_nocs[ids].reshape(-1, 3)

		cloud_xyz_original = cloud_xyz.copy()
		cloud_xyz = NormalizeCloud()(cloud_xyz)

		data_dict = {}
		data_dict['input'] = torch.as_tensor(cloud_xyz.astype(np.float32)).contiguous()
		data_dict['cloud_xyz_original'] = torch.as_tensor(cloud_xyz_original.astype(np.float32)).contiguous()
		data_dict['cloud_nocs'] = torch.as_tensor(cloud_nocs.astype(np.float32)).contiguous()
		return data_dict

	def __getitem__(self, index):
		file = self.files[index]
		while 1:
			try:
				with gzip.open(file, 'rb') as ff:
					data = pickle.load(ff)
				break
			except Exception as e:
				time.sleep(0.001)

		data = self.transform(data)
		return data