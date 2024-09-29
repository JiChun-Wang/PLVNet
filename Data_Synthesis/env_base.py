import os

import pybullet as p
import pybullet_data

import Data_Synthesis.utils as PU


class EnvBase:
	def __init__(self, gui=False):
		if not p.isConnected():
			if gui:
				self.client_id = p.connect(p.GUI)
				p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, rgbBackground=[1, 1, 1])
				# p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
				p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-45,
				                                    cameraTargetPosition=[0, 0, 0])
				code_dir = os.path.dirname(os.path.realpath(__file__))
				self.logging_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, f'{code_dir}/video.mp4')
			else:
				self.client_id = p.connect(p.DIRECT)
		else:
			print('bullet server already connected')

		self.gui = gui
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		code_dir = os.path.dirname(os.path.realpath(__file__))
		p.setAdditionalSearchPath(f'{code_dir}/..')
		self.env_body_ids = {}
		# self.id_to_obj_file = {}
		# self.id_to_scales = {}

	def __del__(self):
		try:
			if self.gui:
				p.stopStateLogging(self.logging_id)
			p.disconnect()
			print("pybullet disconnected")
		except Exception as e:
			pass

	def reset(self):
		body_ids = PU.get_bodies()
		for body_id in body_ids:
			if body_id in self.env_body_ids:
				continue
			p.removeBody(body_id)
		# body_ids = list(self.id_to_obj_file.keys())
		# for body_id in body_ids:
		# 	if body_id in self.env_body_ids:
		# 		continue
		# 	del self.id_to_obj_file[body_id]
		# 	del self.id_to_scales[body_id]
