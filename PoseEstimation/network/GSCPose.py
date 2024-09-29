import numpy as np
import torch
import torch.nn as nn

from .PoseNet import PoseNet_Re, PoseNet
from ..losses.pose_loss import Pose_loss
from ..losses.vote_loss import Vote_line_loss, Vote_plane_loss


class GSCPose_Re(nn.Module):
	def __init__(self, cfg):
		super(GSCPose_Re, self).__init__()
		self.posenet = nn.DataParallel(PoseNet_Re())
		self.loss_pose = Pose_loss(cfg)
		self.loss_vote = Vote_line_loss(cfg)

		self.name_pose_list = ['Rot1', 'Rot1_cos', 'Tran']
		self.name_vote_list = ['Vote_line', 'Vote_line_con']

	def forward(self, points, gt_R, gt_t, do_loss=False, phase='val', aug_rt_t=None, aug_rt_r=None):
		PC = points.detach()
		if phase == 'train':
			PC_da, gt_R_da, gt_t_da = self.data_augment(PC, gt_R, gt_t, aug_rt_t, aug_rt_r)
			PC = PC_da
			gt_R = gt_R_da
			gt_t = gt_t_da
		vote, p_green_R, Pred_T = self.posenet(PC)

		output_dict = {}
		output_dict['vote'] = vote
		output_dict['PC'] = PC
		output_dict['p_green_R'] = p_green_R
		output_dict['p_red_R'] = None
		output_dict['Pred_T'] = Pred_T
		output_dict['gt_R'] = gt_R
		output_dict['gt_t'] = gt_t

		if do_loss:
			p_vote = vote
			p_T = Pred_T

			# pose loss
			pred_pose_list = {
				'Rot1': p_green_R,    # [bs, 3]
				'Tran': p_T,          # [bs, 3]
			}

			# gt_green_v, gt_red_v = get_gt_v(gt_R)
			gt_green_v = gt_R[:, :, 2]
			gt_pose_list = {
				'Rot1': gt_green_v,   # [bs, 3]
				'Tran': gt_t,         # [bs, 3]
			}
			pose_loss = self.loss_pose(self.name_pose_list, pred_pose_list, gt_pose_list)

			# vote loss
			pred_vote_list = {
				'Vote': p_vote,
				'Rot1': p_green_R,
				'Tran': p_T,
			}

			gt_vote_list = {
				'Points': PC,
				'R': gt_R,
				'T': gt_t
			}
			vote_loss = self.loss_vote(self.name_vote_list, pred_vote_list, gt_vote_list)

			loss_dict = {}
			loss_dict['pose_loss'] = pose_loss
			loss_dict['vote_loss'] = vote_loss
		else:
			return output_dict

		return output_dict, loss_dict

	def defor_3D_rt(self, pc, R, t, aug_rt_t, aug_rt_r):
		#  add_t
		dx = aug_rt_t[0]
		dy = aug_rt_t[1]
		dz = aug_rt_t[2]

		pc[:, 0] = pc[:, 0] + dx
		pc[:, 1] = pc[:, 1] + dy
		pc[:, 2] = pc[:, 2] + dz
		t[0] = t[0] + dx
		t[1] = t[1] + dy
		t[2] = t[2] + dz

		Rm = aug_rt_r
		pc_new = torch.mm(Rm, pc.T).T
		pc = pc_new
		R_new = torch.mm(Rm, R)
		R = R_new
		T_new = torch.mm(Rm, t.view(3, 1))
		t = T_new

		return pc, R, t

	def data_augment(self, PC, gt_R, gt_t, aug_rt_t, aug_rt_r):
		# augmentation
		bs = PC.shape[0]
		for i in range(bs):
			prop_rt = torch.rand(1)
			if prop_rt < 0.3:
				PC_new, gt_R_new, gt_t_new = self.defor_3D_rt(PC[i, ...], gt_R[i, ...], gt_t[i, ...], aug_rt_t[i, ...], aug_rt_r[i, ...])
				PC[i, ...] = PC_new
				gt_R[i, ...] = gt_R_new
				gt_t[i, ...] = gt_t_new.view(-1)

		return PC, gt_R, gt_t

	def build_params(self, to_freeze_dict=None):
		#  training_stage is a list that controls whether to freeze each module
		params_lr_list = []

		if to_freeze_dict != None:
			for name, param in self.posenet.named_parameters():
				for freeze_term in to_freeze_dict :
					if freeze_term in name:
						with torch.no_grad():
							param.requires_grad = False

		# pose
		params_lr_list.append(
			{
				"params" : filter(lambda p : p.requires_grad, self.posenet.parameters())}
		)

		return params_lr_list


# class GSCPose(nn.Module):
# 	def __init__(self, cfg):
# 		super(GSCPose, self).__init__()
# 		self.posenet = nn.DataParallel(PoseNet())
# 		self.loss_pose = Pose_loss(cfg)
# 		self.loss_vote = Vote_plane_loss(cfg)
#
# 		# self.name_pose_list = ['Rot1', 'Rot2', 'Rot1_cos', 'Rot2_cos', 'Rot_regular', 'Tran']
# 		self.name_pose_list = ['Rot1', 'Rot2', 'Tran']
# 		self.name_vote_list = ['Vote_plane']
#
# 	def forward(self, points, gt_R, gt_t, do_loss=False, phase='val', aug_rt_t=None, aug_rt_r=None):
# 		PC = points.detach()
# 		if phase == 'train':
# 			PC_da, gt_R_da, gt_t_da = self.data_augment(PC, gt_R, gt_t, aug_rt_t, aug_rt_r)
# 			PC = PC_da
# 			gt_R = gt_R_da
# 			gt_t = gt_t_da
# 		vote, p_green_R, p_red_R, Pred_T = self.posenet(PC)
#
# 		output_dict = {}
# 		output_dict['vote'] = vote
# 		output_dict['PC'] = PC
# 		output_dict['p_green_R'] = p_green_R
# 		output_dict['p_red_R'] = p_red_R
# 		output_dict['Pred_T'] = Pred_T
# 		output_dict['gt_R'] = gt_R
# 		output_dict['gt_t'] = gt_t
#
# 		if do_loss:
# 			p_vote = vote
# 			p_T = Pred_T
#
# 			# pose loss
# 			pred_pose_list = {
# 				'Rot1': p_green_R,    # [bs, 3]
# 				'Rot2': p_red_R,      # [bs, 3]
# 				'Tran': p_T,          # [bs, 3]
# 			}
#
# 			gt_green_v, gt_red_v = gt_R[:, :, 2], gt_R[:, :, 1]
# 			gt_pose_list = {
# 				'Rot1': gt_green_v,   # [bs, 3]
# 				'Rot2': gt_red_v,
# 				'Tran': gt_t,         # [bs, 3]
# 			}
# 			pose_loss = self.loss_pose(self.name_pose_list, pred_pose_list, gt_pose_list)
#
# 			# vote loss
# 			pred_vote_list = {
# 				'Vote': p_vote,
# 				'Rot1': p_green_R,
# 				'Rot2': p_red_R,
# 				'Tran': p_T,
# 			}
#
# 			gt_vote_list = {
# 				'Points': PC,
# 				'R': gt_R,
# 				'T': gt_t
# 			}
# 			vote_loss = self.loss_vote(self.name_vote_list, pred_vote_list, gt_vote_list)
#
# 			loss_dict = {}
# 			loss_dict['pose_loss'] = pose_loss
# 			loss_dict['vote_loss'] = vote_loss
# 		else:
# 			return output_dict
#
# 		return output_dict, loss_dict
#
# 	def defor_3D_rt(self, pc, R, t, aug_rt_t, aug_rt_r):
# 		#  add_t
# 		dx = aug_rt_t[0]
# 		dy = aug_rt_t[1]
# 		dz = aug_rt_t[2]
#
# 		pc[:, 0] = pc[:, 0] + dx
# 		pc[:, 1] = pc[:, 1] + dy
# 		pc[:, 2] = pc[:, 2] + dz
# 		t[0] = t[0] + dx
# 		t[1] = t[1] + dy
# 		t[2] = t[2] + dz
#
# 		Rm = aug_rt_r
# 		pc_new = torch.mm(Rm, pc.T).T
# 		pc = pc_new
# 		R_new = torch.mm(Rm, R)
# 		R = R_new
# 		T_new = torch.mm(Rm, t.view(3, 1))
# 		t = T_new
#
# 		return pc, R, t
#
# 	def data_augment(self, PC, gt_R, gt_t, aug_rt_t, aug_rt_r):
# 		# augmentation
# 		bs = PC.shape[0]
# 		for i in range(bs):
# 			prop_rt = torch.rand(1)
# 			if prop_rt < 0.3:
# 				PC_new, gt_R_new, gt_t_new = self.defor_3D_rt(PC[i, ...], gt_R[i, ...], gt_t[i, ...], aug_rt_t[i, ...], aug_rt_r[i, ...])
# 				PC[i, ...] = PC_new
# 				gt_R[i, ...] = gt_R_new
# 				gt_t[i, ...] = gt_t_new.view(-1)
#
# 		return PC, gt_R, gt_t
#
# 	def build_params(self, to_freeze_dict=None):
# 		#  training_stage is a list that controls whether to freeze each module
# 		params_lr_list = []
#
# 		if to_freeze_dict != None:
# 			for name, param in self.posenet.named_parameters():
# 				for freeze_term in to_freeze_dict :
# 					if freeze_term in name:
# 						with torch.no_grad():
# 							param.requires_grad = False
#
# 		# pose
# 		params_lr_list.append(
# 			{
# 				"params" : filter(lambda p : p.requires_grad, self.posenet.parameters())}
# 		)
#
# 		return params_lr_list


class GSCPose(nn.Module):
	def __init__(self, cfg):
		super(GSCPose, self).__init__()
		# self.sym_type = cfg.sym_type
		self.posenet = nn.DataParallel(PoseNet(cfg.sym_type))
		self.loss_pose = Pose_loss(cfg)
		self.loss_vote = Vote_plane_loss(cfg)

		# self.name_pose_list = ['Rot1', 'Rot2', 'Rot1_cos', 'Rot2_cos', 'Rot_regular', 'Tran']
		if cfg.sym_type == 2:
			self.name_pose_list = ['Rot1', 'Rot2', 'Tran']
		else:
			self.name_pose_list = ['Rot1', 'Tran']
		self.name_vote_list = ['Vote_plane']

	def forward(self, points, gt_R, gt_t, gt_size, do_loss=False, phase='val', aug_rt_t=None, aug_rt_r=None):
		PC = points.detach()
		if phase == 'train':
			PC_da, gt_R_da, gt_t_da = self.data_augment(PC, gt_R, gt_t, aug_rt_t, aug_rt_r)
			PC = PC_da
			gt_R = gt_R_da
			gt_t = gt_t_da
		vote, vote_point1, vote_point2, Pred_T = self.posenet(PC)

		output_dict = {}
		output_dict['vote'] = vote
		output_dict['vote_point1'] = vote_point1
		output_dict['vote_point2'] = vote_point2
		output_dict['PC'] = PC
		output_dict['Pred_T'] = Pred_T
		output_dict['gt_R'] = gt_R
		output_dict['gt_t'] = gt_t

		if do_loss:
			p_vote = vote
			p_T = Pred_T

			# pose loss
			pred_pose_list = {
				'Vote1': vote_point1,  # [bs, n, 3],
				'Vote2': vote_point2,
				'T': p_T,              # [bs, 3]
			}

			gt_pose_list = {
				'Points': PC,
				'R': gt_R,   # [bs, 3]
				'T': gt_t,   # [bs, 3]
				'Size': gt_size
			}
			pose_loss = self.loss_pose(self.name_pose_list, pred_pose_list, gt_pose_list)

			# vote loss
			pred_vote_list = {
				'Vote': p_vote
			}

			gt_vote_list = {
				'Points': PC,
				'R': gt_R,
				'T': gt_t
			}
			vote_loss = self.loss_vote(self.name_vote_list, pred_vote_list, gt_vote_list)

			loss_dict = {}
			loss_dict['pose_loss'] = pose_loss
			loss_dict['vote_loss'] = vote_loss
		else:
			return output_dict

		return output_dict, loss_dict

	def defor_3D_rt(self, pc, R, t, aug_rt_t, aug_rt_r):
		#  add_t
		dx = aug_rt_t[0]
		dy = aug_rt_t[1]
		dz = aug_rt_t[2]

		pc[:, 0] = pc[:, 0] + dx
		pc[:, 1] = pc[:, 1] + dy
		pc[:, 2] = pc[:, 2] + dz
		t[0] = t[0] + dx
		t[1] = t[1] + dy
		t[2] = t[2] + dz

		Rm = aug_rt_r
		pc_new = torch.mm(Rm, pc.T).T
		pc = pc_new
		R_new = torch.mm(Rm, R)
		R = R_new
		T_new = torch.mm(Rm, t.view(3, 1))
		t = T_new

		return pc, R, t

	def data_augment(self, PC, gt_R, gt_t, aug_rt_t, aug_rt_r):
		# augmentation
		bs = PC.shape[0]
		for i in range(bs):
			prop_rt = torch.rand(1)
			if prop_rt < 0.3:
				PC_new, gt_R_new, gt_t_new = self.defor_3D_rt(PC[i, ...], gt_R[i, ...], gt_t[i, ...], aug_rt_t[i, ...], aug_rt_r[i, ...])
				PC[i, ...] = PC_new
				gt_R[i, ...] = gt_R_new
				gt_t[i, ...] = gt_t_new.view(-1)

		return PC, gt_R, gt_t

	def build_params(self, to_freeze_dict=None):
		#  training_stage is a list that controls whether to freeze each module
		params_lr_list = []

		if to_freeze_dict != None:
			for name, param in self.posenet.named_parameters():
				for freeze_term in to_freeze_dict :
					if freeze_term in name:
						with torch.no_grad():
							param.requires_grad = False

		# pose
		params_lr_list.append(
			{
				"params" : filter(lambda p : p.requires_grad, self.posenet.parameters())}
		)

		return params_lr_list