import numpy as np
from transforms3d.euler import euler2mat
import torch
import torch.nn as nn


def get_symmetry_tfs():
	tfs = []
	for rz in [0, np.pi]:
		tf = np.eye(4)
		tf[:3, :3] = euler2mat(0, 0, rz, axes='sxyz')
		tfs.append(tf)
	return np.array(tfs)


def to_homo_torch(pts):
	'''
  @pts: shape can be (B,N,3 or 2) or (N,3) will homogeneliaze the last dimension
  '''
	ones = torch.ones((*pts.shape[:-1], 1)).to(pts.device).float()
	homo = torch.cat((pts, ones), dim=-1)
	return homo


class NocsMinSymmetryCELoss(nn.Module):
	def __init__(self, ce_loss_bins=100):
		super().__init__()
		self.ce_loss_bins = ce_loss_bins
		self.symmetry_tfs = get_symmetry_tfs()
		new_tfs = []
		for symmetry_tf in self.symmetry_tfs:
			tf = torch.from_numpy(symmetry_tf).cuda().float()
			new_tfs.append(tf)
		self.symmetry_tfs = torch.stack(new_tfs, dim=0)
		self.n_sym = len(self.symmetry_tfs)
		self.bin_resolution = 1 / ce_loss_bins

	def forward(self, pred, target):
		"""
		pred: [bs, npts, 3*100]
		target" [bs, npts, 3]
		"""
		B, N = target.shape[:2]
		tmp_target = torch.matmul(self.symmetry_tfs.unsqueeze(0).expand(B, self.n_sym, 4, 4),
		                          to_homo_torch(target - 0.5).permute(0, 2, 1).unsqueeze(1).expand(B, self.n_sym,
		                                                                                           4, -1))
		# tmp_target: [bs, n_tfs, n_pts, 3]
		tmp_target = tmp_target.permute(0, 1, 3, 2)[..., :3] + 0.5
		cloud_nocs_bin_class = torch.clamp(tmp_target / self.bin_resolution, 0, self.ce_loss_bins - 1).long()

		pred = pred.reshape(B, -1, 3, self.ce_loss_bins).unsqueeze(-1).expand(-1, -1, -1, -1, self.n_sym)
		# [bs, n_pts, 3, 100, n_tfs]

		loss = []
		for i in range(3):
			# [bs, n_pts, 100, n_tfs] -> [bs, 100, n_tfs, n_pts]
			loss.append(nn.CrossEntropyLoss(reduction='none')(pred[:, :, i].permute(0, 2, 3, 1),
			                                                  cloud_nocs_bin_class[..., i]))  # [bs, n_tfs, n_pts]
		loss = torch.stack(loss, dim=-1).sum(dim=-1)
		loss = loss.mean(dim=-1)
		ids = loss.argmin(dim=1)
		loss = torch.gather(loss, dim=1, index=ids.unsqueeze(1))
		loss = loss.mean()
		return loss
