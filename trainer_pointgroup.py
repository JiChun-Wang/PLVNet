import os
import time
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from PointGroup.data.dataset_seg import Dataset
from PointGroup.model.pointgroup.pointgroup import PointGroup, model_fn_decorator
from PointGroup.util.config import get_parser
from PointGroup.util.utils import AverageMeter


def init():
	os.makedirs(cfg.exp_dir, exist_ok=True)
	os.system('cp {} {}'.format(cfg.config_seg, cfg.exp_dir))

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


class TrainerPointGroup:
	def __init__(self):
		self.epoch = 1

		self.best_train = 1e9
		self.best_val = 1e9

		self.train_data = Dataset(cfg, phase='train')
		self.val_data = Dataset(cfg, phase='val')

		self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=cfg.batch_size,
		                                                shuffle=True, num_workers=cfg.n_workers,
		                                                pin_memory=False, drop_last=True,
		                                                worker_init_fn=worker_init_fn,
		                                                collate_fn=self.train_data.merge)
		self.val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=cfg.batch_size,
		                                              shuffle=True, num_workers=cfg.n_workers,
		                                              pin_memory=False, drop_last=False,
		                                              worker_init_fn=worker_init_fn,
		                                              collate_fn=self.val_data.merge)

		self.model = PointGroup(cfg)
		# self.model = nn.DataParallel(self.model)
		self.model.cuda()

		if cfg.optim == 'Adam':
			self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.lr)
		elif cfg.optim == 'SGD':
			self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.lr,
			                                 weight_decay=cfg.weight_decay, momentum=cfg.momentum)

		self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=cfg.lr_milestones, gamma=0.1)

	def train_loop(self):
		iter_time = AverageMeter()
		data_time = AverageMeter()
		avg_loss = {}

		model_fn = model_fn_decorator(cfg, test=False)
		self.model.train()
		start_epoch = time.time()
		end = time.time()
		for iter, batch in enumerate(self.train_loader):
			data_time.update(time.time() - end)
			loss, preds, visual_dict, meter_dict = model_fn(batch, self.model, self.epoch)

			for k, v in visual_dict.items():
				if k not in avg_loss.keys():
					avg_loss[k] = AverageMeter()
				avg_loss[k].update(v)

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			iter_time.update(time.time() - end)
			end = time.time()

			current_iter = (self.epoch - 1) * len(self.train_loader) + iter + 1
			max_iter = cfg.epochs * len(self.train_loader)
			remain_iter = max_iter - current_iter

			remain_time = remain_iter * iter_time.avg
			t_m, t_s = divmod(remain_time, 60)
			t_h, t_m = divmod(t_m, 60)
			remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

			print("epoch: {}/{} iter: {}/{} loss: {:.6f} semantic_loss: {:.6f} offset_norm_loss: {:.6f} data_time: {:.2f}({:.2f}) iter_time: {:.2f}({:.2f}) "
			      "remain_time: {remain_time}".format(
				self.epoch, cfg.epochs, iter+1, len(self.train_loader), avg_loss['loss'].val, avg_loss['semantic_loss'].val,
				avg_loss['offset_norm_loss'].val, data_time.val, data_time.avg, iter_time.val,
				iter_time.avg, remain_time=remain_time))

			if (iter == len(self.train_loader) - 1): print()

		print("epoch: {}/{}, train loss: {:.6f}, time: {}s".format(self.epoch, cfg.epochs, avg_loss['loss'].avg,
		                                                      time.time() - start_epoch))

		if self.epoch % cfg.save_freq == 0:
			self.checkpoint_save(phase='train')

		if avg_loss['loss'].avg < self.best_train:
			self.best_train = avg_loss['loss'].avg
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
			save_path = f'{cfg.exp_dir}/pointgroup_{self.epoch:03d}.pth.tar'
		else:
			save_path = f'{cfg.exp_dir}/pointgroup_{f}.pth.tar'

		torch.save(checkpoint_data, save_path, _use_new_zipfile_serialization=False)

	def val_loop(self):
		model_fn = model_fn_decorator(cfg, test=False)
		avg_loss = {}

		with torch.no_grad():
			self.model.eval()
			start_epoch = time.time()
			for iter, batch in enumerate(self.val_loader):
				loss, preds, visual_dict, meter_dict = model_fn(batch, self.model, self.epoch)

				for k, v in visual_dict.items():
					if k not in avg_loss.keys():
						avg_loss[k] = AverageMeter()
					avg_loss[k].update(v)

				# if iter % max(1, len(self.val_loader) // 10) == 0:
				# 	print('epoch={}, {}/{}, val_loss={}'.format(self.epoch, iter, len(self.val_loader), loss.item()))

				print("iter: {}/{} loss: {:.6f}({:.6f})".format(iter+1, len(self.val_loader), avg_loss['loss'].val,
				                                             avg_loss['loss'].avg))
				if (iter == len(self.val_loader) - 1): print()

			print("epoch: {}/{}, val loss: {:.6f}, time: {}s".format(self.epoch, cfg.epochs, avg_loss['loss'].avg,
			                                                    time.time() - start_epoch))

			if avg_loss['loss'].avg < self.best_val:
				self.best_val = avg_loss['loss'].avg
				self.checkpoint_save(phase='val', f='best_val')

		for k in avg_loss.keys():
			writer.add_scalar(f'Val/{k}', avg_loss[k].avg, self.epoch)

	def train(self):
		for self.epoch in range(cfg.epochs+1):
			print(f'>>>>>>>>>> Training Epoch: {self.epoch}/{cfg.epochs} <<<<<<<<<<')
			self.train_loop()
			print(">>>>>>>>>> End <<<<<<<<<<")

			print(f'>>>>>>>>>> Start Evaluation <<<<<<<<<<')
			self.val_loop()
			print(">>>>>>>>>> End <<<<<<<<<<")

			self.scheduler.step()
			torch.cuda.empty_cache()


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

	trainer = TrainerPointGroup()
	trainer.train()
