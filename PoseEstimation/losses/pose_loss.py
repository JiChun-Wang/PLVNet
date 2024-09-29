import torch
import torch.nn as nn

# from ..util.config import cfg # can control the weight of each term here


# class Pose_loss(nn.Module):
# 	def __init__(self, cfg):
# 		self.cfg = cfg
# 		super(Pose_loss, self).__init__()
# 		if cfg.pose_loss_type == 'l1':
# 			self.loss_func_t = nn.L1Loss()
# 			self.loss_func_s = nn.L1Loss()
# 			self.loss_func_Rot = nn.L1Loss()
# 		elif cfg.pose_loss_type == 'smoothl1':   # same as MSE
# 			self.loss_func_t = nn.SmoothL1Loss(beta=0.5)
# 			self.loss_func_s = nn.SmoothL1Loss(beta=0.5)
# 			self.loss_func_Rot = nn.SmoothL1Loss(beta=0.5)
# 		else:
# 			raise NotImplementedError
#
# 	def forward(self, name_list, pred_list, gt_list):
# 		loss_list = {}
# 		if "Rot1" in name_list:
# 			loss_list["Rot1"] = self.cfg.rot_1_w * self.cal_loss_Rot1(pred_list["Rot1"], gt_list["Rot1"])
# 		else:
# 			loss_list["Rot1"] = 0.0
#
# 		if "Rot1_cos" in name_list:
# 			loss_list["Rot1_cos"] = self.cfg.rot_1_w * self.cal_cosine_dis1(pred_list["Rot1"], gt_list["Rot1"])
# 		else:
# 			loss_list["Rot1_cos"] = 0.0
#
# 		if "Rot2" in name_list:
# 			loss_list["Rot2"] = self.cfg.rot_2_w * self.cal_loss_Rot2(pred_list["Rot2"], gt_list["Rot2"])
# 		else:
# 			loss_list["Rot2"] = 0.0
#
# 		if "Rot2_cos" in name_list:
# 			loss_list["Rot2_cos"] = self.cfg.rot_2_w * self.cal_cosine_dis2(pred_list["Rot2"], gt_list["Rot2"])
# 		else:
# 			loss_list["Rot2_cos"] = 0.0
#
# 		if "Rot_regular" in name_list:
# 			loss_list["Rot_regular"] = self.cfg.rot_regular_w * self.cal_rot_regular_angle(pred_list["Rot1"],
# 																				  pred_list["Rot2"],)
# 		else:
# 			loss_list["Rot_regular"] = 0.0
#
# 		if "Tran" in name_list:
# 			loss_list["Tran"] = self.cfg.tran_w * self.cal_loss_Tran(pred_list["Tran"], gt_list["Tran"])
# 		else:
# 			loss_list["Tran"] = 0.0
#
# 		if "Size" in name_list:
# 			loss_list["Size"] = self.cfg.size_w * self.cal_loss_Size(pred_list["Size"], gt_list["Size"])
# 		else:
# 			loss_list["Size"] = 0.0
#
# 		return loss_list
#
# 	def cal_loss_Rot1(self, pred_v, gt_v):
# 		bs = pred_v.shape[0]
# 		res = torch.zeros([bs], dtype=torch.float32, device=pred_v.device)
# 		for i in range(bs):
# 			pred_v_now = pred_v[i, ...]
# 			gt_v_now = gt_v[i, ...]
# 			# if self.cfg.obj_name == 'tless_29':
# 			# 	res[i] = torch.min(self.loss_func_Rot(pred_v_now, gt_v_now), self.loss_func_Rot(pred_v_now, -gt_v_now))
# 			# else :
# 			# 	res[i] = self.loss_func_Rot(pred_v_now, gt_v_now)
# 			res[i] = self.loss_func_Rot(pred_v_now, gt_v_now)
# 		res = torch.mean(res)
# 		return res
#
# 	def cal_cosine_dis1(self, pred_v, gt_v):
# 		# pred_v  bs x 3, gt_v bs x 3
# 		bs = pred_v.shape[0]
# 		res = torch.zeros([bs], dtype=torch.float32).to(pred_v.device)
# 		for i in range(bs):
# 			pred_v_now = pred_v[i, ...]
# 			gt_v_now = gt_v[i, ...]
# 			# if self.cfg.obj_name == 'tless_29':
# 			# 	res[i] = torch.min((1.0 - torch.sum(pred_v_now * gt_v_now)) * 2.0,
# 			# 	                   (1.0 - torch.sum(pred_v_now * -gt_v_now)) * 2.0)
# 			# else :
# 			# 	res[i] = (1.0 - torch.sum(pred_v_now * gt_v_now)) * 2.0
# 			res[i] = (1.0 - torch.sum(pred_v_now * gt_v_now)) * 2.0
# 		res = torch.mean(res)
# 		return res
#
# 	def cal_loss_Rot2(self, pred_v, gt_v):
# 		bs = pred_v.shape[0]
# 		res = torch.zeros([bs], dtype=torch.float32, device=pred_v.device)
# 		for i in range(bs):
# 			pred_v_now = pred_v[i, ...]
# 			gt_v_now = gt_v[i, ...]
# 			if self.cfg.sym_type == 1:
# 				res[i] = torch.min(self.loss_func_Rot(pred_v_now, gt_v_now), self.loss_func_Rot(pred_v_now, -gt_v_now))
# 			else:
# 				res[i] = self.loss_func_Rot(pred_v_now, gt_v_now)
# 		res = torch.mean(res)
# 		return res
#
# 	def cal_cosine_dis2(self, pred_v, gt_v):
# 		# pred_v  bs x 3, gt_v bs x 3
# 		bs = pred_v.shape[0]
# 		res = torch.zeros([bs], dtype=torch.float32).to(pred_v.device)
# 		for i in range(bs):
# 			pred_v_now = pred_v[i, ...]
# 			gt_v_now = gt_v[i, ...]
# 			if self.cfg.sym_type == 1:
# 				res[i] = torch.min((1.0 - torch.sum(pred_v_now * gt_v_now)) * 2.0,
# 				                   (1.0 - torch.sum(pred_v_now * -gt_v_now)) * 2.0)
# 			else:
# 				res[i] = (1.0 - torch.sum(pred_v_now * gt_v_now)) * 2.0
# 		res = torch.mean(res)
# 		return res
#
# 	def cal_rot_regular_angle(self, pred_v1, pred_v2):
# 		bs = pred_v1.shape[0]
# 		res = torch.zeros([bs], dtype=torch.float32).to(pred_v1.device)
# 		for i in range(bs):
# 			z_direction = pred_v1[i, ...]
# 			x_direction = pred_v2[i, ...]
# 			res[i] = torch.abs(torch.dot(z_direction, x_direction))
# 		res = torch.mean(res)
# 		return res
#
# 	def cal_loss_Tran(self, pred_trans, gt_trans):
# 		return self.loss_func_t(pred_trans, gt_trans)
#
# 	def cal_loss_Size(self, pred_size, gt_size):
# 		return self.loss_func_s(pred_size, gt_size)


class Pose_loss(nn.Module):
	def __init__(self, cfg):
		self.cfg = cfg
		super(Pose_loss, self).__init__()
		if cfg.pose_loss_type == 'l1':
			self.loss_func_t = nn.L1Loss()
			self.loss_func_s = nn.L1Loss()
			self.loss_func_Rot = nn.L1Loss()
		elif cfg.pose_loss_type == 'smoothl1':   # same as MSE
			self.loss_func_t = nn.SmoothL1Loss(beta=0.5)
			self.loss_func_s = nn.SmoothL1Loss(beta=0.5)
			self.loss_func_Rot = nn.SmoothL1Loss(beta=0.5)
		else:
			raise NotImplementedError

	def forward(self, name_list, pred_list, gt_list):
		loss_list = {}
		if "Rot1" in name_list:
			loss_list["Rot1"] = self.cfg.rot_1_w * self.cal_loss_Rot1(gt_list['Points'],
			                                                          pred_list["Vote1"],
			                                                          gt_list["R"],
			                                                          gt_list["T"],
			                                                          gt_list["Size"])
		else:
			loss_list["Rot1"] = 0.0

		if "Rot1_cos" in name_list:
			loss_list["Rot1_cos"] = self.cfg.rot_1_w * self.cal_cosine_dis1(pred_list["Rot1"], gt_list["Rot1"])
		else:
			loss_list["Rot1_cos"] = 0.0

		if "Rot2" in name_list:
			loss_list["Rot2"] = self.cfg.rot_2_w * self.cal_loss_Rot2(gt_list['Points'],
			                                                          pred_list["Vote2"],
			                                                          gt_list["R"],
			                                                          gt_list["T"],
			                                                          gt_list["Size"])
		else:
			loss_list["Rot2"] = 0.0

		if "Rot2_cos" in name_list:
			loss_list["Rot2_cos"] = self.cfg.rot_2_w * self.cal_cosine_dis2(pred_list["Rot2"], gt_list["Rot2"])
		else:
			loss_list["Rot2_cos"] = 0.0

		if "Rot_regular" in name_list:
			loss_list["Rot_regular"] = self.cfg.rot_regular_w * self.cal_rot_regular_angle(pred_list["Rot1"],
																				  pred_list["Rot2"],)
		else:
			loss_list["Rot_regular"] = 0.0

		if "Tran" in name_list:
			loss_list["Tran"] = self.cfg.tran_w * self.cal_loss_Tran(pred_list["T"], gt_list["T"])
		else:
			loss_list["Tran"] = 0.0

		return loss_list

	def cal_loss_Rot1(self, PC, PC_vote, gt_R, gt_t, gt_s):
		bs, n_pts = PC.shape[0], PC.shape[1]
		res_p_vote = 0.0

		for i in range(bs):
			PC_now = PC[i, ...]
			PC_vote_now = PC_vote[i, ...]
			p_s_now = gt_s[i, ...]

			kpt_cano = torch.tensor([0, 0, p_s_now[2]], dtype=torch.float32).to(PC.device)
			kpt = torch.mm(gt_R[i, ...], kpt_cano.view(-1, 1)) + gt_t[i, ...].view(-1, 1)
			kpt = kpt.view(1, 3)
			gt_vote = kpt - PC_now
			res_p_vote += self.loss_func_Rot(PC_vote_now, gt_vote)

		res_p_vote = res_p_vote / bs
		return res_p_vote

	def cal_loss_Rot2(self, PC, PC_vote, gt_R, gt_t, gt_s):
		bs, n_pts = PC.shape[0], PC.shape[1]
		res_p_vote = 0.0

		for i in range(bs):
			PC_now = PC[i, ...]
			PC_vote_now = PC_vote[i, ...]
			p_s_now = gt_s[i, ...]

			kpt_cano = torch.tensor([0, p_s_now[1], 0], dtype=torch.float32).to(PC.device)
			kpt = torch.mm(gt_R[i, ...], kpt_cano.view(-1, 1)) + gt_t[i, ...].view(-1, 1)
			kpt = kpt.view(1, 3)
			gt_vote = kpt - PC_now
			res_p_vote += self.loss_func_Rot(PC_vote_now, gt_vote)

		res_p_vote = res_p_vote / bs
		return res_p_vote

	def cal_loss_Tran(self, pred_trans, gt_trans):
		return self.loss_func_t(pred_trans, gt_trans)
