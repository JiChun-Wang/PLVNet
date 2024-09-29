import torch.nn as nn
import torch
import torch.nn.functional as F

# from ..util.config import cfg

# Point_center  encode the segmented point cloud
# one more conv layer compared to original paper

# class Pose_T(nn.Module):
# 	def __init__(self):
# 		super(Pose_T, self).__init__()
# 		self.f = cfg.feat_c_t
# 		self.k = cfg.t_c
#
# 		self.conv1 = torch.nn.Conv1d(self.f, 1024, 1)
#
# 		self.conv2 = torch.nn.Conv1d(1024, 256, 1)
# 		self.conv3 = torch.nn.Conv1d(256, 256, 1)
# 		self.conv4 = torch.nn.Conv1d(256, self.k, 1)
# 		self.drop1 = nn.Dropout(0.2)
# 		self.bn1 = nn.BatchNorm1d(1024)
# 		self.bn2 = nn.BatchNorm1d(256)
# 		self.bn3 = nn.BatchNorm1d(256)
#
# 	def forward(self, x):
# 		x = F.relu(self.bn1(self.conv1(x)))
# 		x = F.relu(self.bn2(self.conv2(x)))
#
# 		x = torch.max(x, 2, keepdim=True)[0]
#
# 		x = F.relu(self.bn3(self.conv3(x)))
# 		x = self.drop1(x)
# 		x = self.conv4(x)
#
# 		x = x.squeeze(2)
# 		x = x.contiguous()
# 		return x


class Point_center(nn.Module):
	def __init__(self):
		super(Point_center, self).__init__()

		self.conv1 = torch.nn.Conv1d(3, 128, 1)
		self.conv2 = torch.nn.Conv1d(128, 256, 1)
		self.conv3 = torch.nn.Conv1d(256, 512, 1)

		self.bn1 = nn.BatchNorm1d(128)
		self.bn2 = nn.BatchNorm1d(256)
		self.bn3 = nn.BatchNorm1d(512)

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = (self.bn3(self.conv3(x)))
		x = torch.max(x, -1, keepdim=True)[0]
		return x


class Pose_T_v2(nn.Module):
	def __init__(self):
		super(Pose_T_v2, self).__init__()

		self.feat = Point_center()
		self.conv1 = torch.nn.Conv1d(512, 256, 1)
		self.conv2 = torch.nn.Conv1d(256, 128, 1)
		self.conv3 = torch.nn.Conv1d(128, 3, 1)

		self.bn1 = nn.BatchNorm1d(256)
		self.bn2 = nn.BatchNorm1d(128)
		self.drop1 = nn.Dropout(0.2)

	def forward(self, x):
		x = x.permute(0, 2, 1)
		x = self.feat(x)

		x = F.relu(self.bn1(self.conv1(x)))
		x = self.bn2(self.conv2(x))

		x=self.drop1(x)
		x = self.conv3(x)

		x = x.squeeze(2)
		x = x.contiguous()
		return x


class Pose_Ts(nn.Module):
	def __init__(self):
		super(Pose_Ts, self).__init__()

		self.feat = Point_center()
		self.conv1 = torch.nn.Conv1d(512, 256, 1)
		self.conv2 = torch.nn.Conv1d(256, 128, 1)
		self.conv3 = torch.nn.Conv1d(128, 6, 1)

		self.bn1 = nn.BatchNorm1d(256)
		self.bn2 = nn.BatchNorm1d(128)
		self.drop1 = nn.Dropout(0.2)

	def forward(self, x):
		x = x.permute(0, 2, 1)
		x = self.feat(x)

		x = F.relu(self.bn1(self.conv1(x)))
		x = self.bn2(self.conv2(x))

		x=self.drop1(x)
		x = self.conv3(x)

		x = x.squeeze(2)
		x = x.contiguous()
		xt = x[:, 0:3]
		xs = x[:, 3:6]
		return xt, xs
