import numpy as np

import torch


def estimate3DLine_worker(cur_pts, points, PassThreshold):
	line = cur_pts[1] - cur_pts[0]
	line = line / (torch.norm(line) + 1e-6)

	pc_t_res = points - cur_pts[0].view(1, -1)
	vec_along_line = torch.mm(torch.mm(pc_t_res, line.view(-1, 1)), line.view(1, -1))
	vec_a_2_b = pc_t_res - vec_along_line

	errs = torch.norm(vec_a_2_b, dim=-1)
	ratio = torch.sum(errs <= PassThreshold) / len(errs)
	inliers = torch.where(errs <= PassThreshold)[0]

	return ratio, line, inliers


def estimate3DLine(points, PassThreshold, max_iter=1000, translation=None):
	pts_lst = []
	ids = np.random.choice(len(points), size=max_iter, replace=False)
	for i in range(max_iter):
		cur_pts = torch.concat([translation.view(1, 3), points[ids[i]].view(1, 3)], dim=0)
		pts_lst.append(cur_pts)

	outs = []
	for i in range(len(pts_lst)):
		out = estimate3DLine_worker(pts_lst[i], points, PassThreshold)
		outs.append((out))

	ratios = []
	lines = []
	inlierss = []
	for out in outs:
		ratio, line, inliers = out
		ratios.append(ratio)
		lines.append(line)
		inlierss.append(inliers)

	best_id = torch.tensor(ratios).argmax()
	best_ratio = ratios[best_id]
	best_line = lines[best_id]
	inliers = inlierss[best_id]
	return best_ratio, best_line, inliers


def linear_fitting_3D_points(points):
	"""
	fitting 3D points to a line which is parameterized by:
		x = k1 * z + b1
		y = k2 * z + b2
	:param points: [n, 3]
	:return:
	"""
	n = len(points)
	Xs, Ys, Zs = points[:, 0], points[:, 1], points[:, 2]
	Sum_X = Xs.sum()
	Sum_Y = Ys.sum()
	Sum_Z = Zs.sum()
	Sum_XZ = (Xs * Zs).sum()
	Sum_YZ = (Ys * Zs).sum()
	Sum_ZZ = (Zs * Zs).sum()

	den = n * Sum_ZZ - Sum_Z * Sum_Z
	k1 = (n * Sum_XZ - Sum_X * Sum_Z) / den
	k2 = (n * Sum_YZ - Sum_Y * Sum_Z) / den

	line = torch.tensor([k1, k2, 1.]).to(points.device)
	line = line / (torch.norm(line) + 1e-6)
	return line