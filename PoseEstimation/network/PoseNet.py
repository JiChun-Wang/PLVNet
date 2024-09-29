import torch
import torch.nn as nn

from .PoseR import Rot_red, Rot_green
from .PoseT import Pose_T_v2, Pose_Ts
from .GeoVote import GeoVote


class PoseNet_Re(nn.Module):
	def __init__(self,):
		super(PoseNet_Re, self).__init__()
		self.geo_vote = GeoVote(sym_type=0)
		self.rot_green = Rot_green()
		self.trans = Pose_T_v2()

	def forward(self, points):
		vote, feat = self.geo_vote(points - points.mean(dim=1, keepdim=True))  # b x n x 3

		#  rotation (normalization)
		green_R_vec = self.rot_green(feat.permute(0, 2, 1))  # b x 3
		p_green_R = green_R_vec / (torch.norm(green_R_vec, dim=1, keepdim=True) + 1e-6)
		# translation
		# feat_for_ts = torch.cat([feat, points-points.mean(dim=1, keepdim=True)], dim=2)
		# T = self.trans(feat_for_ts.permute(0, 2, 1))
		T = self.trans(points - points.mean(dim=1, keepdim=True))
		Pred_T = T + points.mean(dim=1)  # bs x 3

		return vote, p_green_R, Pred_T


# class PoseNet(nn.Module):
# 	def __init__(self):
# 		super(PoseNet, self).__init__()
# 		self.geo_vote = GeoVote(sym_type=1)
# 		self.rot_green = Rot_green()
# 		self.rot_red = Rot_red()
# 		self.trans = Pose_T_v2()
#
# 	def forward(self, points):
# 		vote, feat = self.geo_vote(points - points.mean(dim=1, keepdim=True))
#
# 		#  rotation
# 		green_R_vec = self.rot_green(feat.permute(0, 2, 1))  # b x 3
# 		red_R_vec = self.rot_red(feat.permute(0, 2, 1))   # b x 3
# 		# normalization
# 		p_green_R = green_R_vec / (torch.norm(green_R_vec, dim=1, keepdim=True) + 1e-6)
# 		p_red_R = red_R_vec / (torch.norm(red_R_vec, dim=1, keepdim=True) + 1e-6)
# 		# translation
# 		# feat_for_ts = torch.cat([feat, points-points.mean(dim=1, keepdim=True)], dim=2)
# 		# T = self.trans(feat_for_ts.permute(0, 2, 1))
# 		T = self.trans(points - points.mean(dim=1, keepdim=True))
# 		Pred_T = T + points.mean(dim=1)  # bs x 3
#
# 		return vote, p_green_R, p_red_R, Pred_T


# class PoseNet(nn.Module):
# 	def __init__(self):
# 		super(PoseNet, self).__init__()
# 		self.geo_vote = GeoVote(sym_type=1)
# 		self.rot_green = Rot_green()
# 		self.trans = Pose_T_v2()
#
# 	def forward(self, points):
# 		vote_line, feat = self.geo_vote(points - points.mean(dim=1, keepdim=True))
#
# 		# vote for point
# 		vote_point1 = self.rot_green(feat.permute(0, 2, 1))  # b x 3
# 		vote_point1 = vote_point1.permute(0, 2, 1)
#
# 		T = self.trans(points - points.mean(dim=1, keepdim=True))
# 		Pred_T = T + points.mean(dim=1)  # bs x 3
#
# 		return vote_line, vote_point1, Pred_T


class PoseNet(nn.Module):
	def __init__(self, sym_type):
		super(PoseNet, self).__init__()
		self.sym_type = sym_type
		self.geo_vote = GeoVote(sym_type)
		self.rot_green = Rot_green()
		if sym_type == 2:
			self.rot_red = Rot_red()
		self.trans = Pose_T_v2()

	def forward(self, points):
		vote_line, feat = self.geo_vote(points - points.mean(dim=1, keepdim=True))

		T = self.trans(points - points.mean(dim=1, keepdim=True))
		Pred_T = T + points.mean(dim=1)  # bs x 3

		# vote for point
		vote_point1 = self.rot_green(feat.permute(0, 2, 1))
		vote_point1 = vote_point1.permute(0, 2, 1)            # b x n x 3
		if self.sym_type == 2:
			vote_point2 = self.rot_red(feat.permute(0, 2, 1))
			vote_point2 = vote_point2.permute(0, 2, 1)        # b x n x 3
			return  vote_line, vote_point1, vote_point2, Pred_T

		return vote_line, vote_point1, None, Pred_T


# class CatPoseNet(nn.Module):
# 	def __init__(self):
# 		super(CatPoseNet, self).__init__()
# 		self.geo_vote = GeoVote(sym_type=1)
# 		self.rot_green = Rot_green()
# 		self.rot_red = Rot_red()
# 		self.ts = Pose_Ts()
#
# 	def forward(self, points):
# 		vote, feat = self.geo_vote(points - points.mean(dim=1, keepdim=True))
#
# 		#  rotation
# 		green_R_vec = self.rot_green(feat.permute(0, 2, 1))  # b x 3
# 		red_R_vec = self.rot_red(feat.permute(0, 2, 1))   # b x 3
# 		# normalization
# 		p_green_R = green_R_vec / (torch.norm(green_R_vec, dim=1, keepdim=True) + 1e-6)
# 		p_red_R = red_R_vec / (torch.norm(red_R_vec, dim=1, keepdim=True) + 1e-6)
# 		# translation
# 		# feat_for_ts = torch.cat([feat, points-points.mean(dim=1, keepdim=True)], dim=2)
# 		# T = self.trans(feat_for_ts.permute(0, 2, 1))
# 		T, s = self.ts(points - points.mean(dim=1, keepdim=True))
# 		Pred_T = T + points.mean(dim=1)  # bs x 3
# 		Pred_s = s
#
# 		return vote, p_green_R, p_red_R, Pred_T, Pred_s


class CatPoseNet(nn.Module):
	def __init__(self, vote_type=0):
		super(CatPoseNet, self).__init__()
		self.vote_type = vote_type
		self.geo_vote = GeoVote(sym_type=1)

		self.rot_green = Rot_green()
		self.rot_red = Rot_red()
		self.ts = Pose_Ts()

	def forward(self, points):
		bs, p_num = points.shape[0], points.shape[1]
		vote, feat = self.geo_vote(points - points.mean(dim=1, keepdim=True))

		#  rotation
		green_R_vec = self.rot_green(feat.permute(0, 2, 1))  # b x 3
		red_R_vec = self.rot_red(feat.permute(0, 2, 1))   # b x 3
		p_green_R = green_R_vec / (torch.norm(green_R_vec, dim=1, keepdim=True) + 1e-6)
		p_red_R = red_R_vec / (torch.norm(red_R_vec, dim=1, keepdim=True) + 1e-6)

		# translation and size
		T, s = self.ts(points - points.mean(dim=1, keepdim=True))
		Pred_T = T + points.mean(dim=1)  # bs x 3
		Pred_s = s

		# vote
		if self.vote_type == 1:
			vote[:, :, :3] += points.mean(dim=1, keepdim=True)
			vote[:, :, 3:] += points.mean(dim=1, keepdim=True)

		return vote, p_green_R, p_red_R, Pred_T, Pred_s