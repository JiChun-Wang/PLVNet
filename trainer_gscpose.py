import os
import time
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from PoseEstimation.datasets.dataset_pose import PoseDataset, CatPoseDataset
from PoseEstimation.network.GSCPose import GSCPose_Re, GSCPose
from PoseEstimation.network.CatPose import CatGSCPose
from PoseEstimation.util.config import get_parser
from PoseEstimation.util.utils import AverageMeter


def init():
	os.makedirs(cfg.exp_dir, exist_ok=True)
	os.system('cp {} {}'.format(cfg.config_pose, cfg.exp_dir))

	# summary writer
	global writer
	writer = SummaryWriter(cfg.exp_dir)

	# random seed
	random.seed(cfg.manual_seed)
	np.random.seed(cfg.manual_seed)
	torch.manual_seed(cfg.manual_seed)
	torch.cuda.manual_seed_all(cfg.manual_seed)


def worker_init_fn(worker_id):
	np.random.seed(np.random.get_state()[1][0] + worker_id)


class TrainerPoseEstimation:
	def __init__(self, cfg):
		self.cfg = cfg
		self.epoch = 1

		self.best_train = 1e9
		self.best_val = 1e9

		self.train_data = PoseDataset(cfg, phase='train')
		self.val_data = PoseDataset(cfg, phase='val')

		self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=cfg.batch_size,
		                                                shuffle=True, num_workers=cfg.n_workers,
		                                                pin_memory=False, drop_last=True,
		                                                worker_init_fn=worker_init_fn)
		self.val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=cfg.batch_size,
		                                              shuffle=True, num_workers=cfg.n_workers,
		                                              pin_memory=False, drop_last=False,
		                                              worker_init_fn=worker_init_fn)

		# if cfg.sym_type == 0:
		# 	self.model = GSCPose_Re(cfg)
		# else:
		# 	self.model = GSCPose(cfg)
		self.model = GSCPose(cfg)

		cfg.resume = None
		if cfg.resume != None:
			checkpoint = torch.load(f'{cfg.exp_dir}/{cfg.resume}')
			print('Pose Estimation: loading model checkpoint from epoch {}'.format(checkpoint['epoch']))
			self.model.load_state_dict(checkpoint['state_dict'])
		# self.model = nn.DataParallel(self.model)
		self.model.cuda()

		param_lists = self.model.build_params()
		if cfg.optim == 'Adam':
			self.optimizer = torch.optim.Adam(param_lists, lr=cfg.lr)
		elif cfg.optim == 'SGD':
			self.optimizer = torch.optim.SGD(param_lists, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)

		self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=cfg.lr_milestones, gamma=cfg.lr_factor)

	def train_loop(self):
		iter_time = AverageMeter()
		data_time = AverageMeter()
		avg_loss = {}

		self.model.train()
		start_epoch = time.time()
		end = time.time()
		for iter, batch in enumerate(self.train_loader):
			data_time.update(time.time() - end)
			output_dict, loss_dict = self.model(
				batch['points'].cuda(),
				batch['Rs'].cuda(),
				batch['ts'].cuda(),
				batch['size'].cuda(),
				do_loss=True,
				phase='train',
				aug_rt_t=batch['aug_rt_t'].cuda(),
				aug_rt_r=batch['aug_rt_R'].cuda(),
			)

			pose_loss = loss_dict['pose_loss']
			vote_loss = loss_dict['vote_loss']
			total_loss = sum(pose_loss.values()) + sum(vote_loss.values())

			self.optimizer.zero_grad()
			total_loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
			self.optimizer.step()

			visual_dict = {'Total': total_loss.item()}
			for k, v in pose_loss.items():
				visual_dict[k] = v.item() if type(v) == torch.Tensor else v
			for k, v in vote_loss.items():
				visual_dict[k] = v.item() if type(v) == torch.Tensor else v

			for k, v in visual_dict.items():
				if k not in avg_loss.keys():
					avg_loss[k] = AverageMeter()
				avg_loss[k].update(v)

			iter_time.update(time.time() - end)
			end = time.time()

			current_iter = (self.epoch - 1) * len(self.train_loader) + iter + 1
			max_iter = cfg.epochs * len(self.train_loader)
			remain_iter = max_iter - current_iter

			remain_time = remain_iter * iter_time.avg
			t_m, t_s = divmod(remain_time, 60)
			t_h, t_m = divmod(t_m, 60)
			remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

			# if cfg.sym_type == 0 :
			# 	print("epoch: {}/{} iter: {}/{} "
			# 	      "Total: {:.6f} Rot1: {:.6f} Rot1_cos: {:.6f} Tran: {:.6f} Vote_line: {:.6f} Vote_line_con: {:.6f} "
			# 	      "data_time: {:.2f}({:.2f}) iter_time: {:.2f}({:.2f}) remain_time: {remain_time}".format(
			# 		      self.epoch, cfg.epochs, iter + 1, len(self.train_loader),
			# 		      avg_loss['Total'].val, avg_loss['Rot1'].val, avg_loss['Rot1_cos'].val,
			# 		      avg_loss['Tran'].val,
			# 		      avg_loss['Vote_line'].val, avg_loss['Vote_line_con'].val,
			# 		      data_time.val, data_time.avg, iter_time.val, iter_time.avg, remain_time=remain_time))
			# else:
			# 	print("epoch: {}/{} iter: {}/{} "
			# 	      "Total: {:.6f} Rot1: {:.6f} Rot1_cos: {:.6f}  Rot2: {:.6f} Rot2_cos: {:.6f} Rot_regular: {:.6f} Tran: {:.6f} "
			# 	      "Vote_plane: {:.6f} Vote_plane_con: {:.6f} "
			# 	      "data_time: {:.2f}({:.2f}) iter_time: {:.2f}({:.2f}) remain_time: {remain_time}".format(
			# 		      self.epoch, cfg.epochs, iter + 1, len(self.train_loader),
			# 		      avg_loss['Total'].val, avg_loss['Rot1'].val, avg_loss['Rot1_cos'].val,
			# 		      avg_loss['Rot2'].val, avg_loss['Rot2_cos'].val, avg_loss['Rot_regular'].val,
			# 		      avg_loss['Tran'].val,
			# 		      avg_loss['Vote_plane'].val, avg_loss['Vote_plane_con'].val,
			# 		      data_time.val, data_time.avg, iter_time.val, iter_time.avg, remain_time=remain_time))

			print("epoch: {}/{} iter: {}/{} "
			      "Total: {:.6f} Rot1: {:.6f} Rot1_cos: {:.6f}  Rot2: {:.6f} Rot2_cos: {:.6f} Rot_regular: {:.6f} Tran: {:.6f} "
			      "Vote_plane: {:.6f} Vote_plane_con: {:.6f} "
			      "data_time: {:.2f}({:.2f}) iter_time: {:.2f}({:.2f}) remain_time: {remain_time}".format(
				self.epoch, cfg.epochs, iter + 1, len(self.train_loader),
				avg_loss['Total'].val, avg_loss['Rot1'].val, avg_loss['Rot1_cos'].val,
				avg_loss['Rot2'].val, avg_loss['Rot2_cos'].val, avg_loss['Rot_regular'].val,
				avg_loss['Tran'].val,
				avg_loss['Vote_plane'].val, avg_loss['Vote_plane_con'].val,
				data_time.val, data_time.avg, iter_time.val, iter_time.avg, remain_time=remain_time))

			if (iter == len(self.train_loader) - 1): print()

		print("epoch: {}/{}, train loss: {:.6f}, time: {}s".format(self.epoch, cfg.epochs, avg_loss['Total'].avg,
		                                                      time.time() - start_epoch))

		if self.epoch % cfg.save_freq == 0:
			self.checkpoint_save(phase='train')

		if avg_loss['Total'].avg < self.best_train:
			self.best_train = avg_loss['Total'].avg
			self.checkpoint_save(phase='train', f='best_train')

		for k in avg_loss.keys():
			writer.add_scalar(f'Train/{k}', avg_loss[k].avg, self.epoch)

	def checkpoint_save(self, phase, f=''):
		assert phase in ['train', 'val']
		checkpoint_data = {
			'epoch': self.epoch,
			'state_dict': self.model.state_dict(),
			'best_res': self.best_train if phase == 'train' else self.best_val
		}

		if len(f) == 0:
			save_path = f'{cfg.exp_dir}/gscpose_{self.epoch:03d}.pth.tar'
		else:
			save_path = f'{cfg.exp_dir}/gscpose_{f}.pth.tar'

		torch.save(checkpoint_data, save_path, _use_new_zipfile_serialization=False)

	def val_loop(self):
		avg_loss = {}

		with torch.no_grad():
			self.model.eval()
			start_epoch = time.time()
			for iter, batch in enumerate(self.val_loader):
				output_dict, loss_dict = self.model(
					batch['points'].cuda(),
					batch['Rs'].cuda(),
					batch['ts'].cuda(),
					batch['size'].cuda(),
					do_loss=True,
					phase='val'
				)

				pose_loss = loss_dict['pose_loss']
				vote_loss = loss_dict['vote_loss']
				total_loss = sum(pose_loss.values()) + sum(vote_loss.values())

				visual_dict = {'Total' : total_loss.item()}
				for k, v in pose_loss.items():
					visual_dict[k] = v.item() if type(v) == torch.Tensor else v
				for k, v in vote_loss.items():
					visual_dict[k] = v.item() if type(v) == torch.Tensor else v

				for k, v in visual_dict.items():
					if k not in avg_loss.keys():
						avg_loss[k] = AverageMeter()
					avg_loss[k].update(v)

				print("iter: {}/{} loss: {:.6f}({:.6f})".format(iter+1, len(self.val_loader), avg_loss['Total'].val,
				                                             avg_loss['Total'].avg))
				if (iter == len(self.val_loader) - 1): print()

			print("epoch: {}/{}, val loss: {:.6f}, time: {}s".format(self.epoch, cfg.epochs, avg_loss['Total'].avg,
			                                                    time.time() - start_epoch))

			if avg_loss['Total'].avg < self.best_val:
				self.best_val = avg_loss['Total'].avg
				self.checkpoint_save(phase='val', f='best_val')

		for k in avg_loss.keys():
			writer.add_scalar(f'Val/{k}', avg_loss[k].avg, self.epoch)

	def train(self):
		start_epoch = 1
		for self.epoch in range(start_epoch, cfg.epochs+start_epoch):
			print(f'>>>>>>>>>> Training Epoch: {self.epoch}/{cfg.epochs+start_epoch} <<<<<<<<<<')
			self.train_loop()
			print(">>>>>>>>>> End <<<<<<<<<<")

			print(f'>>>>>>>>>> Start Evaluation <<<<<<<<<<')
			self.val_loop()
			print(">>>>>>>>>> End <<<<<<<<<<")

			self.scheduler.step()


if __name__ == "__main__":
	# parser = argparse.ArgumentParser()
	# parser.add_argument('--dataset', type=str, default='Sileance_Dataset', help='indicate dataset name')
	# parser.add_argument('--obj_name', type=str, default='gear', help='indicate object name')
	# args = parser.parse_args()
	# dataset = args.dataset
	# obj_name = args.obj_name
	cfg = get_parser()
	init()
	print(cfg)

	trainer = TrainerPoseEstimation(cfg)
	trainer.train()
