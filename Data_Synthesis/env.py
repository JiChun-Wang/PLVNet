import os
import random
import time
import copy
import pickle

import numpy as np
from PIL import Image
import cv2
import trimesh
from transforms3d.quaternions import quat2mat, mat2quat
from transformations import random_rotation_matrix
import pybullet as p

import Data_Synthesis.utils as PU
from Data_Synthesis.env_base import EnvBase
from Data_Synthesis.camera import Camera


class Env(EnvBase):
	def __init__(self, bin_model_pth, cam_model_pth, cam_params, gui=False):
		EnvBase.__init__(self, gui)
		self.cam_cfg = cam_params
		self.K = np.array([[cam_params['fu'],         0.      , cam_params['cu']],
		                   [       0.       , cam_params['fv'], cam_params['cv']],
		                   [       0.       ,         0.      ,        1.      ]])
		self.H = cam_params['height']
		self.W = cam_params['width']
		quat = cam_params['rotation']
		tra = cam_params['location']
		self.add_camera(self.H, self.W, cam_model_pth, quat, tra)

		self.bin_in_world = np.eye(4)
		self.add_bin(bin_model_pth)

		p.setGravity(0, 0, -10)

		self.env_body_ids = PU.get_bodies()
		print("self.env_body_ids", self.env_body_ids)

	def add_camera(self, H, W, cam_model, quat, tra):
		self.camera = Camera(self.K, H, W, cam_model)
		self.cam_in_world = np.eye(4)
		self.cam_in_world[:3, :3] = quat2mat(quat)
		self.cam_in_world[:3, 3] = [0, 0, 0.75]
		p.changeVisualShape(self.camera.cam_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
		PU.set_body_pose_in_world(self.camera.cam_id, self.cam_in_world)

	def add_bin(self, obj_file, scale=1.0):
		ob_in_world = self.bin_in_world.copy()
		self.bin_id = PU.create_object(obj_file, scale=[1, 1, 0.5], ob_in_world=ob_in_world, mass=0.1,
		                               useFixedBase=True, concave=True)[0]
		p.changeDynamics(self.bin_id, -1, collisionMargin=0.0001)
		p.changeVisualShape(self.bin_id, -1, rgbaColor=[0.8, 0.8, 0.8, 1])
		# self.id_to_obj_file[self.bin_id] = obj_file
		# self.id_to_scales[self.bin_id] = np.ones((3), dtype=float) * scale
		bin_verts = p.getMeshData(self.bin_id)[1]
		bin_verts = np.array(bin_verts).reshape(-1, 3)
		self.bin_dimensions = bin_verts[:, :2].max(axis=0) - bin_verts[:, :2].min(axis=0)
		self.bin_in_world = PU.get_ob_pose_in_world(self.bin_id)
		self.bin_verts = bin_verts

	def add_duplicate_object_on_pile(self, obj_file, n_ob):
		'''
		@scale: (3) array
		'''
		ob_in_worlds = []
		bin_pose = PU.get_ob_pose_in_world(self.bin_id)
		for i in range(n_ob):
			ob_x = np.random.uniform(-self.bin_dimensions[0] / 2, self.bin_dimensions[0] / 2) + bin_pose[0, 3]
			ob_y = np.random.uniform(-self.bin_dimensions[1] / 2, self.bin_dimensions[1] / 2) + bin_pose[1, 3]
			ob_z = np.random.uniform(0.05, 1) + bin_pose[2, 3]
			ob_pos = np.array([ob_x, ob_y, ob_z])
			R = random_rotation_matrix(np.random.rand(3))

			ob_in_world = np.eye(4)
			ob_in_world[:3, 3] = ob_pos
			ob_in_world[:3, :3] = R[:3, :3]
			ob_in_worlds.append(ob_in_world)

		scale = np.ones((3,), dtype=np.float32)
		ob_ids = PU.create_duplicate_object(n_ob, obj_file, scale, ob_in_worlds, mass=0.1, has_collision=True, concave=False)
		for ob_id in ob_ids:
			# self.id_to_obj_file[ob_id] = copy.deepcopy(obj_file)
			# self.id_to_scales[ob_id] = scale.copy()
			rgba = list(np.random.rand(4))
			rgba[3] = 1.
			p.changeVisualShape(ob_id, -1, rgbaColor=rgba)
			p.changeDynamics(ob_id, -1, linearDamping=0.9, angularDamping=0.9, lateralFriction=0.9, spinningFriction=0.9,
			                 collisionMargin=0.0001)
		return ob_ids

	def simulation_until_stable(self):
		print('simulation_until_stable....')
		n_step = 0
		while 1:
			bin_in_world = PU.get_ob_pose_in_world(self.bin_id)
			for body_id in PU.get_bodies():
				if body_id in self.env_body_ids:
					continue
				ob_in_world = PU.get_ob_pose_in_world(body_id)
				ob_in_bin = np.linalg.inv(bin_in_world) @ ob_in_world
				if ob_in_bin[2, 3] <= -0.02 or np.abs(ob_in_bin[0, 3]) > 0.5 or np.abs(
						ob_in_bin[1, 3]) > 0.5:  # Out of bin
					p.removeBody(body_id)

			last_poses = {}
			accum_motions = {}
			for body_id in PU.get_bodies():
				if body_id in self.env_body_ids:
					continue
				last_poses[body_id] = PU.get_ob_pose_in_world(body_id)
				accum_motions[body_id] = 0

			stabled = True
			for _ in range(50):
				p.stepSimulation()
				n_step += 1
				for body_id in PU.get_bodies():
					if body_id in self.env_body_ids:
						continue
					cur_pose = PU.get_ob_pose_in_world(body_id)
					motion = np.linalg.norm(cur_pose[:3, 3] - last_poses[body_id][:3, 3])
					accum_motions[body_id] += motion
					last_poses[body_id] = cur_pose.copy()
					if accum_motions[body_id] >= 0.001:
						stabled = False
						break
				if stabled == False:
					break

			if stabled:
				for body_id in PU.get_bodies():
					if body_id in self.env_body_ids:
						continue
					p.resetBaseVelocity(body_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
				break

		print('Finished simulation')

	def make_pile(self, obj_file, num_objects):
		# print('Making pile {} scale={}'.format(obj_file, scale))
		body_ids = PU.get_bodies()
		for body_id in body_ids:
			p.changeDynamics(body_id, -1, activationState=p.ACTIVATION_STATE_ENABLE_SLEEPING)
		print("Add new objects on pile #={}".format(num_objects))

		before_ids = PU.get_bodies()
		while 1:
			new_ids = list(set(PU.get_bodies()) - set(before_ids))
			if len(new_ids) == num_objects:
				break
			if len(new_ids) > num_objects:
				to_remove_ids = np.random.choice(np.array(new_ids), size=len(new_ids) - num_objects, replace=False)
				for id in to_remove_ids:
					p.removeBody(id)
				self.simulation_until_stable()
				continue
			self.add_duplicate_object_on_pile(obj_file, num_objects - len(new_ids))
			self.simulation_until_stable()

		#########!NOTE Replace with concave model. Dont do this when making pile, it's too slow
		# assert os.path.exists(obj_file.replace('.obj', '_vhacd2.obj')), f'there is no obj collision file!'
		# ob_concave_urdf_dir = obj_file.replace('.obj', '_vhacd2.urdf')
		# PU.create_urdf_for_mesh(obj_file, out_dir=ob_concave_urdf_dir, concave=True, scale=np.ones((3)))
		#
		# body_ids = PU.get_bodies()
		# for body_id in body_ids:
		# 	if body_id in self.env_body_ids:
		# 		continue
		# 	ob_in_world = PU.get_ob_pose_in_world(body_id)
		# 	p.removeBody(body_id)
		# 	ob_id = p.loadURDF(ob_concave_urdf_dir, [0, 0, 0], useFixedBase=False, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
		# 	PU.set_body_pose_in_world(ob_id, ob_in_world)
		# 	p.changeDynamics(ob_id, -1, linearDamping=0.9, angularDamping=0.9, lateralFriction=0.9,
		# 	                 spinningFriction=0.9, collisionMargin=0.0001,
		# 	                 activationState=p.ACTIVATION_STATE_ENABLE_SLEEPING)
		#
		# self.simulation_until_stable()
		#########!NOTE Replace with concave model. Dont do this when making pile, it's too slow

		self.ob_ids = list(set(PU.get_bodies()) - set(before_ids))

	def generate_one(self, obj_file, num_inst, data_id, out_dir):
		begin = time.time()

		self.make_pile(obj_file, num_inst)

		n_trial = 0
		while 1:
			n_trial += 1
			if n_trial >= 5:
				self.reset()
				self.generate_one(obj_file, num_inst, data_id, out_dir)
				return

			# bin_pose = PU.get_ob_pose_in_world(self.bin_id)
			# self.cam_in_world = bin_pose @ self.cam_in_bin

			rgb, depth, seg = self.camera.render(self.cam_in_world)
			seg[seg < 0] = 0
			if seg.max() >= 65535:
				raise RuntimeError('seg.max={} reaches uint16 limit'.format(seg.max()))

			seg_ids = np.unique(seg)
			if len(set(seg_ids) - set(self.env_body_ids)) == 0:
				print(f'Need continue. seg_ids={seg_ids}, self.env_body_ids={self.env_body_ids}')
				continue

			break

		rgb_dir = '{}/{:05d}rgb.png'.format(out_dir, data_id)
		Image.fromarray(rgb).save(rgb_dir)
		cv2.imwrite(rgb_dir.replace('rgb', 'depth'), (depth * 1000).astype(np.uint16))
		cv2.imwrite(rgb_dir.replace('rgb', 'seg'), seg.astype(np.uint16))
		poses = {}
		for body_id in PU.get_bodies():
			if body_id in self.env_body_ids:
				continue
			poses[body_id] = PU.get_ob_pose_in_world(body_id)

		with open(rgb_dir.replace('rgb.png', 'meta.pkl'), 'wb') as ff:
			# meta = {'cam_in_world': self.cam_in_world, 'K': self.K, 'id_to_obj_file': self.id_to_obj_file, 'poses': poses,
			#         'id_to_scales': self.id_to_scales, 'env_body_ids': self.env_body_ids}
			meta = {'cam_in_world': self.cam_in_world,
			        'K': self.K,
			        'H': self.H,
			        'W': self.W,
			        'poses': poses,
			        'env_body_ids': self.env_body_ids}
			pickle.dump(meta, ff)
		print("Saved to {}".format(rgb_dir))

		print("Generate one sample time {} s".format(time.time() - begin))
		print(">>>>>>>>>>>>>>>>>>>>>>")

	def generate_one_window(self, obj_file, num_inst, data_id, out_dir):
		begin = time.time()
		self.make_pile(obj_file, num_inst)

		n_trial = 0
		while 1:
			n_trial += 1
			if n_trial >= 5:
				self.reset()
				poses = self.generate_one_window(obj_file, num_inst, data_id, out_dir)
				return poses

			rgb, depth, seg = self.camera.render(self.cam_in_world)
			seg[seg < 0] = 0
			if seg.max() >= 65535:
				raise RuntimeError('seg.max={} reaches uint16 limit'.format(seg.max()))

			seg_ids = np.unique(seg)
			if len(set(seg_ids) - set(self.env_body_ids)) == 0:
				print(f'Need continue. seg_ids={seg_ids}, self.env_body_ids={self.env_body_ids}')
				continue

			break

		poses = {}
		for body_id in PU.get_bodies():
			if body_id in self.env_body_ids:
				continue
			poses[body_id] = PU.get_ob_pose_in_world(body_id)

		rgb_dir = '{}/{:05d}rgb.png'.format(out_dir, data_id)
		Image.fromarray(rgb).save(rgb_dir)
		Image.fromarray((depth * 1000).astype(np.uint16)).save(rgb_dir.replace('rgb', 'depth'))
		Image.fromarray(seg.astype(np.uint16)).save(rgb_dir.replace('rgb', 'seg'))
		with open(rgb_dir.replace('rgb.png', 'meta.pkl'), 'wb') as ff:
			meta = {'cam_in_world': self.cam_in_world,
			        'K': self.K,
			        'H': self.H,
			        'W': self.W,
			        'poses': poses,
			        'env_body_ids': self.env_body_ids}
			pickle.dump(meta, ff)

		print("Generate one sample time {} s".format(time.time() - begin))
		return poses


useRealTimeSimulation = 1 # 0 will freeze the simualtion?
TIMESTEP_ = 1. / 240. # Time in seconds.


class Env_v2(EnvBase):
	def __init__(self, bin_model_pth, cam_model_pth, cam_params, gui=False):
		EnvBase.__init__(self, gui)
		self.cam_cfg = cam_params
		self.K = np.array([[cam_params['fu'],         0.      , cam_params['cu']],
		                   [       0.       , cam_params['fv'], cam_params['cv']],
		                   [       0.       ,         0.      ,        1.      ]])
		self.H = cam_params['height']
		self.W = cam_params['width']
		quat = cam_params['rotation']
		tra = cam_params['location']
		self.add_camera(self.H, self.W, cam_model_pth, quat, tra)

		self.bin_in_world = np.eye(4)
		self.add_bin(bin_model_pth)

		p.setGravity(0, 0, -10)
		if useRealTimeSimulation:
			p.setRealTimeSimulation(1)

		self.env_body_ids = PU.get_bodies()
		print("self.env_body_ids", self.env_body_ids)

	def add_camera(self, H, W, cam_model, quat, tra):
		self.camera = Camera(self.K, H, W, cam_model)
		self.cam_in_world = np.eye(4)
		self.cam_in_world[:3, :3] = quat2mat(quat)
		self.cam_in_world[:3, 3] = tra
		PU.set_body_pose_in_world(self.camera.cam_id, self.cam_in_world)

	def add_bin(self, obj_file, scale=1.0):
		ob_in_world = self.bin_in_world.copy()
		self.bin_id = PU.create_object(obj_file, scale=np.ones((3)) * scale, ob_in_world=ob_in_world, mass=0.1,
		                               useFixedBase=True, concave=True)[0]
		bin_verts = p.getMeshData(self.bin_id)[1]
		bin_verts = np.array(bin_verts).reshape(-1, 3)
		self.bin_dimensions = bin_verts[:, :2].max(axis=0) - bin_verts[:, :2].min(axis=0)
		self.bin_in_world = PU.get_ob_pose_in_world(self.bin_id)

		self.drop_x_min = -self.bin_dimensions[0] * 0.75 / 2 + self.bin_in_world[0, 3]
		self.drop_x_max = self.bin_dimensions[0] * 0.75 / 2 + self.bin_in_world[0, 3]
		self.drop_y_min = -self.bin_dimensions[1] * 0.75 / 2 + self.bin_in_world[1, 3]
		self.drop_y_max = self.bin_dimensions[1] * 0.75 / 2 + self.bin_in_world[1, 3]
		self.drop_z_min = 1.5
		self.drop_z_max = 1.0

	def generate_one(self, obj_file, num_inst, data_id, out_dir):
		begin = time.time()
		urdf_file = obj_file.replace('.obj', '.urdf')
		assert os.path.exists(urdf_file), 'There is not a urdf file!'
		obj_id = []
		count = 0
		for i in range(num_inst):
			position = [
				random.uniform(self.drop_x_min, self.drop_x_max),
				random.uniform(self.drop_y_min, self.drop_y_max),
				random.uniform(self.drop_z_min, self.drop_z_max)
			]

			orientation = p.getQuaternionFromEuler([
				random.uniform(0.01, 3.0142),
				random.uniform(0.01, 3.0142),
				random.uniform(0.01, 3.0142)
			])

			obj_id.append(p.loadURDF(urdf_file, position, orientation))
			time.sleep(0.25)  # to prevent all objects drop at the same time
			count += 1
			rgb, depth, seg = self.camera.render(self.cam_in_world)
			if (useRealTimeSimulation) :
				time.sleep(TIMESTEP_)
			else :
				p.stepSimulation()

		time_start = time.time()
		while time.time() < (time_start + 3.0) :
			rgb, depth, seg = self.camera.render(self.cam_in_world)
			if (useRealTimeSimulation) :
				time.sleep(TIMESTEP_)
			else :
				p.stepSimulation()

		rgb_dir = '{}/{:05d}rgb.png'.format(out_dir, data_id)
		Image.fromarray(rgb).save(rgb_dir)
		cv2.imwrite(rgb_dir.replace('rgb', 'depth'), (depth * 1000).astype(np.uint16))
		cv2.imwrite(rgb_dir.replace('rgb', 'seg'), seg.astype(np.uint16))
		poses = {}
		for body_id in PU.get_bodies():
			if body_id in self.env_body_ids:
				continue
			poses[body_id] = PU.get_ob_pose_in_world(body_id)

		with open(rgb_dir.replace('rgb.png', 'meta.pkl'), 'wb') as ff:
			meta = {'cam_in_world': self.cam_in_world,
			        'K': self.K,
			        'H': self.H,
			        'W': self.W,
			        'poses': poses,
			        'env_body_ids': self.env_body_ids}
			pickle.dump(meta, ff)
		print("Saved to {}".format(rgb_dir))

		print("Generate one sample time {} s".format(time.time() - begin))
		print(">>>>>>>>>>>>>>>>>>>>>>")