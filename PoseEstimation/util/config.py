import argparse
import yaml
import os
code_dir = os.path.dirname(os.path.realpath(__file__))


def get_parser(cfg_file=f'{code_dir}/../config/config_pose.yaml'):
	parser = argparse.ArgumentParser(description='Point Cloud Segmentation')
	# parser.add_argument('--config_pose', type=str, default=f'{code_dir}/../config/config_pose.yaml', help='path to config file')
	args_cfg = parser.parse_args()

	# assert args_cfg.config_pose is not None

	setattr(args_cfg, 'config_pose', cfg_file)
	with open(args_cfg.config_pose, 'r') as f:
		config = yaml.safe_load(f)
	for key in config:
		for k, v in config[key].items():
			setattr(args_cfg, k, v)

	setattr(args_cfg, 'exp_dir', os.path.join('PoseEstimation/exp', args_cfg.dataset, args_cfg.obj_name))
	return args_cfg

# cfg = get_parser()
# setattr(cfg, 'exp_dir', os.path.join('PoseEstimation/exp', cfg.dataset, cfg.obj_name))
