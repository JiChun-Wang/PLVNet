import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet2_utils import PointNetSetAbstraction, Upsample
import PoseEstimation.network.gcn3d as gcn3d


class GeoVote(nn.Module):
	def __init__(self, sym_type=0):
		super(GeoVote, self).__init__()
		if sym_type == 0:
			self.vote_num = 3
		else:
			self.vote_num = 6

		self.sa1 = PointNetSetAbstraction(1024, 0.01, 32, 3 + 3, [32, 32, 64], False)
		self.sa2 = PointNetSetAbstraction(256, 0.02, 32, 64 + 3, [64, 64, 128], False)
		self.sa3 = PointNetSetAbstraction(64, 0.04, 32, 128 + 3, [128, 128, 256], False)
		self.sa4 = PointNetSetAbstraction(16, 0.08, 32, 256 + 3, [256, 256, 512], False)

		dim_fuse = sum([64, 128, 256, 512])
		self.vote_head = nn.Sequential(
			nn.Conv1d(dim_fuse, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 256, 1),
			nn.BatchNorm1d(256),
			nn.ReLU(inplace=True),
			nn.Conv1d(256, 128, 1),
			nn.BatchNorm1d(128),
			nn.ReLU(inplace=True),
			nn.Conv1d(128, self.vote_num, 1)
		)

	def forward(self, xyz):
		l0_feats = xyz.permute(0, 2, 1)                           # [bs, 3, 4096]
		l0_points = l0_feats[:, :3, :]                            # [bs, 3, 4096]

		l1_points, l1_feats = self.sa1(l0_points, l0_feats)   # [bs, 3, 1024]   [bs, 64, 1024]
		l2_points, l2_feats = self.sa2(l1_points, l1_feats)   # [bs, 3, 256]    [bs, 128, 256]
		l3_points, l3_feats = self.sa3(l2_points, l2_feats)   # [bs, 3, 64]     [bs, 256, 64]
		l4_points, l4_feats = self.sa4(l3_points, l3_feats)   # [bs, 3, 16]     [bs, 512, 16]

		l4_up_feats = Upsample(l0_points, l4_points, l4_feats)  # [bs, 512, 4096]
		l3_up_feats = Upsample(l0_points, l3_points, l3_feats)  # [bs, 256, 4096]
		l2_up_feats = Upsample(l0_points, l2_points, l2_feats)  # [bs, 128, 4096]
		l1_up_feats = Upsample(l0_points, l1_points, l1_feats)  # [bs, 64, 4096]

		fused_feats = torch.cat([l1_up_feats, l2_up_feats, l3_up_feats, l4_up_feats], dim=1)   # [bs, 960, 4096]
		vote = self.vote_head(fused_feats)
		return vote.permute(0, 2, 1), fused_feats.permute(0, 2, 1)


# class GeoVote(nn.Module):
# 	def __init__(self, sym_type=0):
# 		super(GeoVote, self).__init__()
# 		self.neighbor_num = 10
# 		self.support_num = 7
# 		if sym_type == 0:
# 			self.vote_num = 3
# 		else:
# 			self.vote_num = 6
#
# 		# 3D convolution for point cloud
# 		self.conv_0 = gcn3d.Conv_surface(kernel_num=128, support_num=self.support_num)
# 		self.conv_1 = gcn3d.Conv_layer(128, 128, support_num=self.support_num)
# 		self.pool_1 = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
# 		self.conv_2 = gcn3d.Conv_layer(128, 256, support_num=self.support_num)
# 		self.conv_3 = gcn3d.Conv_layer(256, 256, support_num=self.support_num)
# 		self.pool_2 = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
# 		self.conv_4 = gcn3d.Conv_layer(256, 512, support_num=self.support_num)
#
# 		self.bn1 = nn.BatchNorm1d(128)
# 		self.bn2 = nn.BatchNorm1d(256)
# 		self.bn3 = nn.BatchNorm1d(256)
#
# 		dim_fuse = sum([128, 128, 256, 256, 512, 512])
# 		self.vote_head = nn.Sequential(
# 			nn.Conv1d(dim_fuse, 512, 1),
# 			nn.BatchNorm1d(512),
# 			nn.ReLU(inplace=True),
# 			nn.Conv1d(512, 256, 1),
# 			nn.BatchNorm1d(256),
# 			nn.ReLU(inplace=True),
# 			nn.Conv1d(256, 128, 1),
# 			nn.BatchNorm1d(128),
# 			nn.ReLU(inplace=True),
# 			nn.Conv1d(128, self.vote_num, 1),
# 		)
#
# 	def forward(self, vertices: "tensor (bs, vetice_num, 3)"):
# 		"""
# 		Return: (bs, vertice_num, class_num)
# 		"""
# 		#  concate feature
# 		bs, vertice_num, _ = vertices.size()    # [bs, 1024, 3]
#
# 		neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)     # [bs, 1024, 10]
# 		# ss = time.time()
# 		fm_0 = F.relu(self.conv_0(neighbor_index, vertices), inplace=True)         # [bs, 1024, 128]
#
# 		fm_1 = F.relu(self.bn1(self.conv_1(neighbor_index, vertices, fm_0).transpose(1, 2)).transpose(1, 2),
# 					  inplace=True)                                                # [bs, 1024, 128]
# 		v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)                          # [bs, 256, 3]   [bs, 256, 128]
# 		# neighbor_index = gcn3d.get_neighbor_index(v_pool_1, self.neighbor_num)
# 		neighbor_index = gcn3d.get_neighbor_index(v_pool_1,
# 												  min(self.neighbor_num, v_pool_1.shape[1] // 8))    # [bs, 256, 10]
# 		fm_2 = F.relu(self.bn2(self.conv_2(neighbor_index, v_pool_1, fm_pool_1).transpose(1, 2)).transpose(1, 2),
# 					  inplace=True)                                                                  # [bs, 256, 256]
# 		fm_3 = F.relu(self.bn3(self.conv_3(neighbor_index, v_pool_1, fm_2).transpose(1, 2)).transpose(1, 2),
# 					  inplace=True)                                                                  # [bs, 256, 256]
# 		v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)                           # [bs, 64, 3]   [bs, 64, 256]
# 		# neighbor_index = gcn3d.get_neighbor_index(v_pool_2, self.neighbor_num)
# 		neighbor_index = gcn3d.get_neighbor_index(v_pool_2, min(self.neighbor_num,
# 																v_pool_2.shape[1] // 8))         # [bs, 64, 8]
# 		fm_4 = self.conv_4(neighbor_index, v_pool_2, fm_pool_2)                                  # [bs, 64, 512]
# 		f_global = fm_4.max(1)[0]                                                                # [bs, 512]
#
# 		nearest_pool_1 = gcn3d.get_nearest_index(vertices, v_pool_1)
# 		nearest_pool_2 = gcn3d.get_nearest_index(vertices, v_pool_2)
# 		fm_2 = gcn3d.indexing_neighbor(fm_2, nearest_pool_1).squeeze(2)
# 		fm_3 = gcn3d.indexing_neighbor(fm_3, nearest_pool_1).squeeze(2)
# 		fm_4 = gcn3d.indexing_neighbor(fm_4, nearest_pool_2).squeeze(2)
#
# 		feat = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4], dim=2)
# 		'''
# 		feat_face = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4], dim=2)
# 		feat_face = torch.mean(feat_face, dim=1, keepdim=True)  # bs x 1 x channel
# 		feat_face_re = feat_face.repeat(1, feat.shape[1], 1)
# 		'''
# 		# feat_face_re = self.global_perception_head(feat)  # bs x C x 1
# 		feat_vote_global = f_global.view(bs, 1, f_global.shape[1]).repeat(1, feat.shape[1], 1)
# 		feat_fuse = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4, feat_vote_global], dim=2)
# 		# feat is the extracted per pixel level feature
#
# 		conv1d_input = feat_fuse.permute(0, 2, 1)  # (bs, fuse_ch, vertice_num)
# 		vote = self.vote_head(conv1d_input)
# 		return vote.permute(0, 2, 1), feat
