'''
Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation
RANSAC for Similarity Transformation Estimation

Written by Srinath Sridhar. Modified by Bowen.
'''
import open3d as o3d
import numpy as np
import cv2, yaml
import itertools
from .utils import to_homo, toOpen3dCloud
from scipy.spatial import cKDTree


def estimateAffine3D(source, target, PassThreshold):
	'''
	@source: (N,3)
	'''
	ret, transform, inliers = cv2.estimateAffine3D(source, target, confidence=0.999, ransacThreshold=PassThreshold)
	tmp = np.eye(4)
	tmp[:3] = transform
	transform = tmp
	inliers = np.where(inliers > 0)[0]
	return transform, inliers


def estimate9DTransform_worker(cur_src, cur_dst, source, target, PassThreshold, use_kdtree_for_eval=False,
							   kdtree_eval_resolution=None, max_scale=np.array([99, 99, 99]),
							   min_scale=np.array([0, 0, 0]), max_dimensions=None):
	bad_return = None, None, None, None, None, None
	transform, inliers = estimateAffine3D(source=cur_src, target=cur_dst, PassThreshold=PassThreshold)
	new_transform = transform.copy()
	scales = np.linalg.norm(transform[:3, :3], axis=0)
	if (scales > max_scale).any() or (scales < min_scale).any():
		return bad_return

	R = transform[:3, :3] / scales.reshape(1, 3)
	t = transform[:3, 3]
	u, s, vh = np.linalg.svd(R)

	if s.min() < 0.8 or s.max() > 1.2:
		return bad_return

	R = u @ vh
	if np.linalg.det(R) < 0:
		return bad_return

	new_transform[:3, :3] = R @ np.diag(scales)
	transform = new_transform.copy()

	if max_dimensions is not None:
		cloud_at_canonical = (np.linalg.inv(transform) @ to_homo(target).T).T[:, :3]
		dimensions = cloud_at_canonical.max(axis=0) - cloud_at_canonical.min(axis=0)
		if (dimensions > max_dimensions).any():
			return bad_return

	src_transformed = (transform @ to_homo(source).T).T[:, :3]

	if not use_kdtree_for_eval:
		errs = np.linalg.norm(src_transformed - target, axis=-1)
		ratio = np.sum(errs <= PassThreshold) / len(errs)
		inliers = np.where(errs <= PassThreshold)[0]
	else:
		pcd = toOpen3dCloud(target)
		pcd = pcd.voxel_down_sample(voxel_size=kdtree_eval_resolution)
		kdtree = cKDTree(np.asarray(pcd.points).copy())
		dists1, indices1 = kdtree.query(src_transformed)
		pcd = toOpen3dCloud(src_transformed)
		pcd = pcd.voxel_down_sample(voxel_size=kdtree_eval_resolution)
		kdtree = cKDTree(np.asarray(pcd.points).copy())
		dists2, indices2 = kdtree.query(target)
		errs = np.concatenate((dists1, dists2), axis=0).reshape(-1)
		ratio = np.sum(errs <= PassThreshold) / len(errs)
		inliers = np.where(dists1 <= PassThreshold)[0]

	return ratio, transform, inliers, scales, R, t


def estimate9DTransform(source, target, PassThreshold, max_iter=1000, use_kdtree_for_eval=False,
						kdtree_eval_resolution=None, max_scale=np.array([99, 99, 99]), min_scale=np.array([0, 0, 0]),
						max_dimensions=None):
	best_transform = None
	best_ratio = 0
	inliers = None

	n_iter = 0
	srcs = []
	dsts = []
	for i in range(max_iter):
		ids = np.random.choice(len(source), size=4, replace=False)
		cur_src = source[ids]
		cur_dst = target[ids]
		srcs.append(cur_src)
		dsts.append(cur_dst)

	outs = []
	for i in range(len(srcs)):
		out = estimate9DTransform_worker(srcs[i], dsts[i], source, target, PassThreshold, use_kdtree_for_eval,
										 kdtree_eval_resolution=kdtree_eval_resolution, max_scale=max_scale,
										 min_scale=min_scale, max_dimensions=max_dimensions)
		if out[0] is None:
			continue
		outs.append((out))
	if len(outs) == 0:
		return None, None, None, None, None

	ratios = []
	transforms = []
	inlierss = []
	scales = []
	Rs = []
	ts = []
	for out in outs:
		ratio, transform, inliers, scale, R, t = out
		ratios.append(ratio)
		transforms.append(transform)
		inlierss.append(inliers)
		scales.append(scale)
		Rs.append(R)
		ts.append(t)

	best_id = np.array(ratios).argmax()
	best_transform = transforms[best_id]
	inliers = inlierss[best_id]
	best_scale = scales[best_id]
	best_R = Rs[best_id]
	best_t = ts[best_id]
	return best_transform, inliers, best_scale, best_R, best_t


def estimateSimilarityTransform(source: np.array, target: np.array, verbose=False):
	SourceHom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))
	TargetHom = np.transpose(np.hstack([target, np.ones([source.shape[0], 1])]))

	# Auto-parameter selection based on source-target heuristics
	TargetNorm = np.mean(np.linalg.norm(target, axis=1))
	SourceNorm = np.mean(np.linalg.norm(source, axis=1))
	RatioTS = (TargetNorm / SourceNorm)
	RatioST = (SourceNorm / TargetNorm)
	PassT = RatioST if(RatioST>RatioTS) else RatioTS
	StopT = PassT / 100
	nIter = 100
	if verbose:
		print('Pass threshold: ', PassT)
		print('Stop threshold: ', StopT)
		print('Number of iterations: ', nIter)

	SourceInliersHom, TargetInliersHom, BestInlierRatio = getRANSACInliers(SourceHom, TargetHom, MaxIterations=nIter, PassThreshold=PassT, StopThreshold=StopT)

	if(BestInlierRatio < 0.1):
		print('[ WARN ] - Something is wrong. Small BestInlierRatio: ', BestInlierRatio)
		return None, None, None, None

	Scales, Rotation, Translation, OutTransform = estimateSimilarityUmeyama(SourceInliersHom, TargetInliersHom)

	if verbose:
		print('BestInlierRatio:', BestInlierRatio)
		print('Rotation:\n', Rotation)
		print('Translation:\n', Translation)
		print('Scales:', Scales)

	return Scales, Rotation, Translation, OutTransform

def estimateRestrictedAffineTransform(source: np.array, target: np.array, verbose=False):
	SourceHom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))
	TargetHom = np.transpose(np.hstack([target, np.ones([source.shape[0], 1])]))

	RetVal, AffineTrans, Inliers = cv2.estimateAffine3D(source, target)
	# We assume no shear in the affine matrix and decompose into rotation, non-uniform scales, and translation
	Translation = AffineTrans[:3, 3]
	NUScaleRotMat = AffineTrans[:3, :3]
	# NUScaleRotMat should be the matrix SR, where S is a diagonal scale matrix and R is the rotation matrix (equivalently RS)
	# Let us do the SVD of NUScaleRotMat to obtain R1*S*R2 and then R = R1 * R2
	R1, ScalesSorted, R2 = np.linalg.svd(NUScaleRotMat, full_matrices=True)

	if verbose:
		print('-----------------------------------------------------------------------')
	# Now, the scales are sort in ascending order which is painful because we don't know the x, y, z scales
	# Let's figure that out by evaluating all 6 possible permutations of the scales
	ScalePermutations = list(itertools.permutations(ScalesSorted))
	MinResidual = 1e8
	Scales = ScalePermutations[0]
	OutTransform = np.identity(4)
	Rotation = np.identity(3)
	for ScaleCand in ScalePermutations:
		CurrScale = np.asarray(ScaleCand)
		CurrTransform = np.identity(4)
		CurrRotation = (np.diag(1 / CurrScale) @ NUScaleRotMat).transpose()
		CurrTransform[:3, :3] = np.diag(CurrScale) @ CurrRotation
		CurrTransform[:3, 3] = Translation
		# Residual = evaluateModel(CurrTransform, SourceHom, TargetHom)
		Residual = evaluateModelNonHom(source, target, CurrScale,CurrRotation, Translation)
		if verbose:
			# print('CurrTransform:\n', CurrTransform)
			print('CurrScale:', CurrScale)
			print('Residual:', Residual)
			print('AltRes:', evaluateModelNoThresh(CurrTransform, SourceHom, TargetHom))
		if Residual < MinResidual:
			MinResidual = Residual
			Scales = CurrScale
			Rotation = CurrRotation
			OutTransform = CurrTransform

	if verbose:
		print('Best Scale:', Scales)

	if verbose:
		print('Affine Scales:', Scales)
		print('Affine Translation:', Translation)
		print('Affine Rotation:\n', Rotation)
		print('-----------------------------------------------------------------------')

	return Scales, Rotation, Translation, OutTransform

def getRANSACInliers(SourceHom, TargetHom, MaxIterations=100, PassThreshold=200, StopThreshold=1):
	BestResidual = 1e10
	BestInlierRatio = 0
	BestInlierIdx = np.arange(SourceHom.shape[1])
	for i in range(0, MaxIterations):
		# Pick 5 random (but corresponding) points from source and target
		RandIdx = np.random.randint(SourceHom.shape[1], size=5)
		_, _, _, OutTransform = estimateSimilarityUmeyama(SourceHom[:, RandIdx], TargetHom[:, RandIdx])
		Residual, InlierRatio, InlierIdx = evaluateModel(OutTransform, SourceHom, TargetHom, PassThreshold)
		if Residual < BestResidual:
			BestResidual = Residual
			BestInlierRatio = InlierRatio
			BestInlierIdx = InlierIdx
		if BestResidual < StopThreshold:
			break

		# print('Iteration: ', i)
		# print('Residual: ', Residual)
		# print('Inlier ratio: ', InlierRatio)

	return SourceHom[:, BestInlierIdx], TargetHom[:, BestInlierIdx], BestInlierRatio

def evaluateModel(OutTransform, SourceHom, TargetHom, PassThreshold):
	Diff = TargetHom - np.matmul(OutTransform, SourceHom)
	ResidualVec = np.linalg.norm(Diff[:3, :], axis=0)
	Residual = np.linalg.norm(ResidualVec)
	InlierIdx = np.where(ResidualVec < PassThreshold)
	nInliers = np.count_nonzero(InlierIdx)
	InlierRatio = nInliers / SourceHom.shape[1]
	return Residual, InlierRatio, InlierIdx[0]

def evaluateModelNoThresh(OutTransform, SourceHom, TargetHom):
	Diff = TargetHom - np.matmul(OutTransform, SourceHom)
	ResidualVec = np.linalg.norm(Diff[:3, :], axis=0)
	Residual = np.linalg.norm(ResidualVec)
	return Residual

def evaluateModelNonHom(source, target, Scales, Rotation, Translation):
	RepTrans = np.tile(Translation, (source.shape[0], 1))
	TransSource = (np.diag(Scales) @ Rotation @ source.transpose() + RepTrans.transpose()).transpose()
	Diff = target - TransSource
	ResidualVec = np.linalg.norm(Diff, axis=0)
	Residual = np.linalg.norm(ResidualVec)
	return Residual

def testNonUniformScale(SourceHom, TargetHom):
	OutTransform = np.matmul(TargetHom, np.linalg.pinv(SourceHom))
	ScaledRotation = OutTransform[:3, :3]
	Translation = OutTransform[:3, 3]
	Sx = np.linalg.norm(ScaledRotation[0, :])
	Sy = np.linalg.norm(ScaledRotation[1, :])
	Sz = np.linalg.norm(ScaledRotation[2, :])
	Rotation = np.vstack([ScaledRotation[0, :] / Sx, ScaledRotation[1, :] / Sy, ScaledRotation[2, :] / Sz])
	print('Rotation matrix norm:', np.linalg.norm(Rotation))
	Scales = np.array([Sx, Sy, Sz])

	# # Check
	# Diff = TargetHom - np.matmul(OutTransform, SourceHom)
	# Residual = np.linalg.norm(Diff[:3, :], axis=0)
	return Scales, Rotation, Translation, OutTransform

def estimateSimilarityUmeyama(SourceHom, TargetHom):
	# Copy of original paper is at: http://web.stanford.edu/class/cs273/refs/umeyama.pdf
	SourceCentroid = np.mean(SourceHom[:3, :], axis=1)
	TargetCentroid = np.mean(TargetHom[:3, :], axis=1)
	nPoints = SourceHom.shape[1]

	CenteredSource = SourceHom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
	CenteredTarget = TargetHom[:3, :] - np.tile(TargetCentroid, (nPoints, 1)).transpose()

	CovMatrix = np.matmul(CenteredTarget, np.transpose(CenteredSource)) / nPoints

	if np.isnan(CovMatrix).any():
		print('nPoints:', nPoints)
		print(SourceHom.shape)
		print(TargetHom.shape)
		raise RuntimeError('There are NANs in the input.')

	U, D, Vh = np.linalg.svd(CovMatrix, full_matrices=True)
	d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
	if d:
		D[-1] = -D[-1]
		U[:, -1] = -U[:, -1]

	Rotation = np.matmul(U, Vh).T # Transpose is the one that works

	varP = np.var(SourceHom[:3, :], axis=1).sum()
	ScaleFact = 1/varP * np.sum(D) # scale factor
	Scales = np.array([ScaleFact, ScaleFact, ScaleFact])
	ScaleMatrix = np.diag(Scales)

	Translation = TargetHom[:3, :].mean(axis=1) - SourceHom[:3, :].mean(axis=1).dot(ScaleFact*Rotation)

	OutTransform = np.identity(4)
	OutTransform[:3, :3] = ScaleMatrix @ Rotation
	OutTransform[:3, 3] = Translation

	# # Check
	# Diff = TargetHom - np.matmul(OutTransform, SourceHom)
	# Residual = np.linalg.norm(Diff[:3, :], axis=0)
	return Scales, Rotation, Translation, OutTransform
