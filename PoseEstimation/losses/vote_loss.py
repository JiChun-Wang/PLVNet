import torch
import torch.nn as nn

# from ..util.config import cfg # can control the weight of each term here


class Vote_line_loss(nn.Module):
	def __init__(self, cfg):
		super(Vote_line_loss, self).__init__()
		self.cfg = cfg
		if cfg.pose_loss_type == 'l1':
			self.loss_func = nn.L1Loss()
		elif cfg.pose_loss_type == 'smoothl1':  # same as MSE
			self.loss_func = nn.SmoothL1Loss(beta=0.5)
		else:
			raise NotImplementedError

	def forward(self, namelist, pred_list, gt_list):
		loss_list = {}
		if "Vote_line" in namelist:
			loss_list["Vote_line"] = self.cfg.vote_line_w * self.vote_line_loss(gt_list['Points'],
			                                                               pred_list['Vote'],
			                                                               gt_list['R'],
			                                                               gt_list['T'])
		else:
			loss_list["Vote_line"] = 0.0

		if "Vote_line_con" in namelist and (self.cfg.vote_line_con_w > 0):
			loss_list["Vote_line_con"] = self.cfg.vote_line_con_w * self.vote_line_con_loss(gt_list['Points'],
			                                                     pred_list['Vote'],
			                                                     pred_list['Rot1'],
			                                                     pred_list['Tran'])
		else:
			loss_list["Vote_line_con"] = 0.0

		return loss_list

	def vote_line_loss(self, PC, PC_vote, gt_R, gt_t):
		bs, n_pts = PC.shape[0], PC.shape[1]
		points_cano = torch.bmm(gt_R.permute(0, 2, 1), (PC - gt_t.view(bs, 1, -1)).permute(0, 2, 1))
		points_cano = points_cano.permute(0, 2, 1)
		res_l_vote = 0.0

		points_line_cano = torch.zeros([bs, n_pts, 3], dtype=torch.float32).to(PC.device)
		for i in range(bs):
			PC_now = PC[i, ...]
			PC_vote_now = PC_vote[i, ...]
			PC_cano_now = points_cano[i, ...]

			points_line_cano[i, :, 2] = PC_cano_now[:, 2]
			points_line = torch.mm(gt_R[i, ...], points_line_cano[i].T) + gt_t[i, ...].view(-1, 1)
			points_line = points_line.T  # n_pts X 3
			gt_vote = points_line - PC_now
			res_l_vote += self.loss_func(PC_vote_now, gt_vote)

		res_l_vote = res_l_vote / bs
		return res_l_vote

	def vote_line_con_loss(self, PC, PC_vote, p_g_vec, p_t):
		bs, n_pts = PC.shape[0], PC.shape[1]
		res_l_con = 0.0
		for i in range(bs):
			PC_now = PC[i, ...]
			PC_vote_now = PC_vote[i, ...]
			p_g_now = p_g_vec[i, ...]
			p_t_now = p_t[i, ...]

			# pc_t_res = PC_now - p_t_now.view(1, -1)
			# vec_along_p_g = pc_t_res + PC_vote_now
			# vec_along_p_g  = vec_along_p_g / (torch.norm(vec_along_p_g, dim=-1, keepdim=True) + 1e-6)
			# res_l_con += self.loss_func(vec_along_p_g, p_g_now.view(1, -1).repeat(n_pts, 1))
			# res_l_con += torch.mean(1 - torch.mm(vec_along_p_g, p_g_now.view(-1, 1)))

			pc_t_res = PC_now - p_t_now.view(1, -1)
			vec_along_p_g  = torch.mm(torch.mm(pc_t_res, p_g_now.view(-1, 1)), p_g_now.view(1, -1))
			a_to_1_2_b = vec_along_p_g - pc_t_res
			res_l_con += self.loss_func(a_to_1_2_b, PC_vote_now)

		res_l_con = res_l_con / bs
		return res_l_con


class Vote_plane_loss(nn.Module):
	def __init__(self, cfg):
		super(Vote_plane_loss, self).__init__()
		self.sym_type = cfg.sym_type
		self.cfg = cfg
		if cfg.pose_loss_type == 'l1':
			self.loss_func = nn.L1Loss()
		elif cfg.pose_loss_type == 'smoothl1':  # same as MSE
			self.loss_func = nn.SmoothL1Loss(beta=0.5)
		else:
			raise NotImplementedError

	def forward(self, namelist, pred_list, gt_list):
		loss_list = {}
		if "Vote_plane" in namelist:
			loss_list["Vote_plane"] = self.cfg.vote_plane_w * self.vote_plane_loss(gt_list['Points'],
			                                                                  pred_list['Vote'],
			                                                                  gt_list['R'],
			                                                                  gt_list['T'])
		else:
			loss_list["Vote_plane"] = 0.0

		if "Vote_plane_con" in namelist and (self.cfg.vote_plane_con_w > 0):
			loss_list["Vote_plane_con"] = self.cfg.vote_plane_con_w * self.vote_plane_con_loss(gt_list['Points'],
			                                                       pred_list['Vote'],
			                                                       pred_list['Rot1'],
			                                                       pred_list['Rot2'],
			                                                       pred_list['Tran'])
		else:
			loss_list["Vote_plane_con"] = 0.0

		return loss_list

	def vote_plane_loss(self, PC, PC_vote, gt_R, gt_t):
		bs, n_pts = PC.shape[0], PC.shape[1]
		points_cano = torch.bmm(gt_R.permute(0, 2, 1), (PC - gt_t.view(bs, 1, -1)).permute(0, 2, 1))
		points_cano = points_cano.permute(0, 2, 1)
		res_p_vote = 0.0

		points_line_z_cano = torch.zeros([bs, n_pts, 3], dtype=torch.float32).to(PC.device)
		if self.sym_type != 0:
			points_line_y_cano = torch.zeros([bs, n_pts, 3], dtype=torch.float32).to(PC.device)
		for i in range(bs):
			PC_now = PC[i, ...]
			PC_cano_now = points_cano[i, ...]

			PC_vote_z_now = PC_vote[i, :, :3]
			points_line_z_cano[i, :, 2] = PC_cano_now[:, 2]
			points_line_z = torch.mm(gt_R[i, ...], points_line_z_cano[i].T) + gt_t[i, ...].view(-1, 1)
			points_line_z = points_line_z.T  # n_pts X 3
			gt_vote_z = points_line_z - PC_now
			res_p_vote += self.loss_func(PC_vote_z_now, gt_vote_z)

			if self.sym_type != 0:
				PC_vote_y_now = PC_vote[i, :, 3:]
				points_line_y_cano[i, :, 1] = PC_cano_now[:, 1]
				points_line_y = torch.mm(gt_R[i, ...], points_line_y_cano[i].T) + gt_t[i, ...].view(-1, 1)
				points_line_y = points_line_y.T  # n_pts X 3
				gt_vote_y = points_line_y - PC_now
				res_p_vote += self.loss_func(PC_vote_y_now, gt_vote_y)

		if self.sym_type != 0:
			res_p_vote = res_p_vote / 2 / bs
		else:
			res_p_vote = res_p_vote / bs

		return res_p_vote

	def vote_plane_con_loss(self, PC, PC_vote, p_g_vec, p_r_vec, p_t):
		bs, n_pts = PC.shape[0], PC.shape[1]
		res_p_con = 0.0
		for i in range(bs):
			PC_now = PC[i, ...]
			PC_vote_xy_now = PC_vote[i, :, :3]
			PC_vote_zx_now = PC_vote[i, :, 3:]
			p_g_now = p_g_vec[i, ...]
			p_r_now = p_r_vec[i, ...]
			p_t_now = p_t[i, ...]

			pc_t_res = PC_now - p_t_now.view(1, -1)

			vec_along_p_g = -1 * torch.mm(torch.mm(pc_t_res, p_g_now.view(-1, 1)), p_g_now.view(1, -1))
			if self.vote_type == 0:
				res_p_con += self.loss_func(vec_along_p_g, PC_vote_xy_now)
			else:
				PC_b = pc_t_res + vec_along_p_g
				res_p_con += self.loss_func(PC_b, PC_vote_xy_now)

			vec_along_p_r = -1 * torch.mm(torch.mm(pc_t_res, p_r_now.view(-1, 1)), p_r_now.view(1, -1))
			if self.vote_type == 0:
				res_p_con += self.loss_func(vec_along_p_r, PC_vote_zx_now)
			else:
				PC_b = pc_t_res + vec_along_p_r
				res_p_con += self.loss_func(PC_b, PC_vote_zx_now)

		res_l_con = res_p_con / 2 / bs
		return res_l_con


# class Vote_plane_loss(nn.Module):
# 	def __init__(self, cfg):
# 		super(Vote_plane_loss, self).__init__()
# 		self.vote_type = cfg.vote_type
# 		self.cfg = cfg
# 		if cfg.pose_loss_type == 'l1':
# 			self.loss_func = nn.L1Loss()
# 		elif cfg.pose_loss_type == 'smoothl1':  # same as MSE
# 			self.loss_func = nn.SmoothL1Loss(beta=0.5)
# 		else:
# 			raise NotImplementedError
#
# 	def forward(self, namelist, pred_list, gt_list):
# 		loss_list = {}
# 		if "Vote_plane" in namelist:
# 			loss_list["Vote_plane"] = self.cfg.vote_plane_w * self.vote_plane_loss(gt_list['Points'],
# 			                                                                  pred_list['Vote'],
# 			                                                                  gt_list['R'],
# 			                                                                  gt_list['T'])
# 		else:
# 			loss_list["Vote_plane"] = 0.0
#
# 		if "Vote_plane_con" in namelist and (self.cfg.vote_plane_con_w > 0):
# 			loss_list["Vote_plane_con"] = self.cfg.vote_plane_con_w * self.vote_plane_con_loss(gt_list['Points'],
# 			                                                       pred_list['Vote'],
# 			                                                       pred_list['Rot1'],
# 			                                                       pred_list['Rot2'],
# 			                                                       pred_list['Tran'])
# 		else:
# 			loss_list["Vote_plane_con"] = 0.0
#
# 		return loss_list
#
# 	def vote_plane_loss(self, PC, PC_vote, gt_R, gt_t):
# 		bs, n_pts = PC.shape[0], PC.shape[1]
# 		points_cano = torch.bmm(gt_R.permute(0, 2, 1), (PC - gt_t.view(bs, 1, -1)).permute(0, 2, 1))
# 		points_cano = points_cano.permute(0, 2, 1)
# 		res_p_vote = 0.0
#
# 		points_plane_xy_cano = torch.zeros([bs, n_pts, 3], dtype=torch.float32).to(PC.device)
# 		points_plane_zx_cano = torch.zeros([bs, n_pts, 3], dtype=torch.float32).to(PC.device)
# 		for i in range(bs):
# 			PC_now = PC[i, ...]
# 			PC_vote_xy_now = PC_vote[i, :, :3]
# 			PC_vote_zx_now = PC_vote[i, :, 3:]
# 			PC_cano_now = points_cano[i, ...]
#
# 			points_plane_xy_cano[i, :, :2] = PC_cano_now[:, :2].view(-1, 2)
# 			points_plane_xy = torch.mm(gt_R[i, ...], points_plane_xy_cano[i].T) + gt_t[i, ...].view(-1, 1)
# 			points_plane_xy = points_plane_xy.T  # n_pts X 3
# 			if self.vote_type == 0:
# 				gt_vote_xy = points_plane_xy - PC_now
# 			else:
# 				gt_vote_xy = points_plane_xy
# 			res_p_vote += self.loss_func(PC_vote_xy_now, gt_vote_xy)
#
# 			points_plane_zx_cano[i, :, (0, 2)] = PC_cano_now[:, (0, 2)].view(-1, 2)
# 			points_plane_zx = torch.mm(gt_R[i, ...], points_plane_zx_cano[i].T) + gt_t[i, ...].view(-1, 1)
# 			points_plane_zx = points_plane_zx.T  # n_pts X 3
# 			if self.vote_type == 0:
# 				gt_vote_zx = points_plane_zx - PC_now
# 			else:
# 				gt_vote_zx = points_plane_zx
# 			res_p_vote += self.loss_func(PC_vote_zx_now, gt_vote_zx)
#
# 		res_p_vote = res_p_vote / 2 / bs
# 		return res_p_vote
#
# 	def vote_plane_con_loss(self, PC, PC_vote, p_g_vec, p_r_vec, p_t):
# 		bs, n_pts = PC.shape[0], PC.shape[1]
# 		res_p_con = 0.0
# 		for i in range(bs):
# 			PC_now = PC[i, ...]
# 			PC_vote_xy_now = PC_vote[i, :, :3]
# 			PC_vote_zx_now = PC_vote[i, :, 3:]
# 			p_g_now = p_g_vec[i, ...]
# 			p_r_now = p_r_vec[i, ...]
# 			p_t_now = p_t[i, ...]
#
# 			pc_t_res = PC_now - p_t_now.view(1, -1)
#
# 			vec_along_p_g = -1 * torch.mm(torch.mm(pc_t_res, p_g_now.view(-1, 1)), p_g_now.view(1, -1))
# 			if self.vote_type == 0:
# 				res_p_con += self.loss_func(vec_along_p_g, PC_vote_xy_now)
# 			else:
# 				PC_b = pc_t_res + vec_along_p_g
# 				res_p_con += self.loss_func(PC_b, PC_vote_xy_now)
#
# 			vec_along_p_r = -1 * torch.mm(torch.mm(pc_t_res, p_r_now.view(-1, 1)), p_r_now.view(1, -1))
# 			if self.vote_type == 0:
# 				res_p_con += self.loss_func(vec_along_p_r, PC_vote_zx_now)
# 			else:
# 				PC_b = pc_t_res + vec_along_p_r
# 				res_p_con += self.loss_func(PC_b, PC_vote_zx_now)
#
# 		res_l_con = res_p_con / 2 / bs
# 		return res_l_con

