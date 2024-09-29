import numpy as np
import torch
import torch.nn as nn

from .PoseNet import CatPoseNet
from ..losses.pose_loss import Pose_loss
from ..losses.vote_loss import Vote_line_loss, Vote_plane_loss
from ..util.utils import get_gt_v


class CatGSCPose(nn.Module):
	def __init__(self, cfg):
		super(CatGSCPose, self).__init__()
		self.posenet = nn.DataParallel(CatPoseNet())
		self.loss_pose = Pose_loss(cfg)
		self.loss_vote = Vote_plane_loss(cfg)

		# self.name_pose_list = ['Rot1', 'Rot2', 'Rot1_cos', 'Rot2_cos', 'Rot_regular', 'Tran', 'Size']
		self.name_pose_list = ['Rot1', 'Rot2', 'Tran', 'Size']
		# self.name_pose_list = ['Rot1', 'Rot2', 'Rot_regular', 'Tran', 'Size']
		# self.name_vote_list = ['Vote_plane', 'Vote_plane_con']
		self.name_vote_list = []

	def forward(self, points, gt_R, gt_t, gt_s, mean_shape, do_loss=False, phase='val', aug_rt_t=None, aug_rt_r=None):
		PC = points.detach()
		if phase == 'train':
			PC_da, gt_R_da, gt_t_da, gt_s_da = self.data_augment(PC, gt_R, gt_t, gt_s, mean_shape, aug_rt_t, aug_rt_r)
			PC = PC_da
			gt_R = gt_R_da
			gt_t = gt_t_da
			gt_s = gt_s_da

		vote, p_green_R, p_red_R, Pred_T, Pred_s = self.posenet(PC)

		output_dict = {}
		output_dict['vote'] = vote
		output_dict['PC'] = PC
		output_dict['p_green_R'] = p_green_R
		output_dict['p_red_R'] = p_red_R
		output_dict['Pred_T'] = Pred_T
		output_dict['Pred_s'] = Pred_s
		output_dict['gt_R'] = gt_R
		output_dict['gt_t'] = gt_t
		output_dict['gt_s'] = gt_s

		if do_loss:
			p_T = Pred_T
			p_s = Pred_s

			# pose loss
			pred_pose_list = {
				'Rot1': p_green_R,    # [bs, 3]
				'Rot2': p_red_R,      # [bs, 3]
				'Tran': p_T,          # [bs, 3]
				'Size': p_s,          # [bs, 3]
			}

			gt_green_v, gt_red_v = get_gt_v(gt_R)
			gt_pose_list = {
				'Rot1': gt_green_v,   # [bs, 3]
				'Rot2': gt_red_v,
				'Tran': gt_t,         # [bs, 3]
				'Size': gt_s,         # [bs, 3]
			}
			pose_loss = self.loss_pose(self.name_pose_list, pred_pose_list, gt_pose_list)

			# vote loss
			pred_vote_list = {
				'Vote': vote,
				'Rot1': p_green_R,
				'Rot2': p_red_R,
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

	def defor_3D_bb(self, pc, R, t, s, s_x=(0.8, 1.2), s_y=(0.8, 1.2), s_z=(0.8, 1.2)):
		pc_reproj = torch.mm(R.T, (pc - t.view(1, 3)).T).T  # nn x 3
		aug_bb = torch.rand(3).to(pc.device)
		ex = aug_bb[0] * (s_x[1] - s_x[0]) + s_x[0]
		ey = aug_bb[1] * (s_y[1] - s_y[0]) + s_y[0]
		ez = aug_bb[2] * (s_z[1] - s_z[0]) + s_z[0]

		pc_reproj[:, 0] = pc_reproj[:, 0] * ex
		pc_reproj[:, 1] = pc_reproj[:, 1] * ey
		pc_reproj[:, 2] = pc_reproj[:, 2] * ez
		s[0] = s[0] * ex
		s[1] = s[1] * ey
		s[2] = s[2] * ez
		pc_new = torch.mm(R, pc_reproj.T) + t.view(3, 1)
		pc_new = pc_new.T
		return pc_new, s

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

	def defor_3D_pc(self, pc, r=0.005):
		points_defor = torch.randn(pc.shape).to(pc.device)
		pc = pc + points_defor * r * pc
		return pc

	def data_augment(self, PC, gt_R, gt_t, gt_s, mean_shape, aug_rt_t, aug_rt_r):
		# augmentation
		bs = PC.shape[0]
		for i in range(bs):
			prop_bb = torch.rand(1)
			if prop_bb < 0:
				PC_new, gt_s_new = self.defor_3D_bb(PC[i, ...], gt_R[i, ...], gt_t[i, ...], gt_s[i, ...] + mean_shape[i, ...])
				gt_s_new = gt_s_new - mean_shape[i, ...]
				PC[i, ...] = PC_new
				gt_s[i, ...] = gt_s_new

			prop_rt = torch.rand(1)
			if prop_rt < 0:
				PC_new, gt_R_new, gt_t_new = self.defor_3D_rt(PC[i, ...], gt_R[i, ...],
				                                         gt_t[i, ...], aug_rt_t[i, ...], aug_rt_r[i, ...])
				PC[i, ...] = PC_new
				gt_R[i, ...] = gt_R_new
				gt_t[i, ...] = gt_t_new.view(-1)

			prop_pc = torch.rand(1)
			if prop_pc < 0.8:
				PC_new = self.defor_3D_pc(PC[i, ...], 0.005)
				PC[i, ...] = PC_new
		return PC, gt_R, gt_t, gt_s

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