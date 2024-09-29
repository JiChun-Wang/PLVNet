# Dense Point-wise Line Voting for Robust 6D Pose Estimation in Industrial Bin-Picking
This repository contains authors' implementation of PLVNet: Dense Point-wise Line Voting for Robust 6D Pose Estimation in Industrial Bin-Picking. 
Our implementation is based on PointGroup and GPVPose. We warmly welcome any discussions related to our implementation and our paper. Please feel free to open an issue.

## Introduction
Accurate 6D object pose estimation is crucial for robotic grasping and manipulation, particularly in industrial bin-picking scenarios. 
Despite challenges posed by heavy occlusion and object symmetries, we propose a novel approach based on point cloud inputs. 
Our method consists of two stages: first, a modified 3D-UNet is employed for instance segmentation, incorporating visibility prediction to mitigate occlusion effects. 
Subsequently, a point-wise line voting network is introduced to regress offset vectors towards object axis lines, aided by a RANSAC-based line fitting technique for robust pose estimation. 
Experiments on public datasets and real-world environments demonstrate the effectiveness of our approach in accurately estimating poses of objects in industrial bin-picking scenes, outperforming several baselines.

## Environment setup
+ Setup the python environtment
```
conda create -n plvnet python=3.7
conda activate plvnet
 
# install torch 1.8.1 built from cuda 11.1
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

pip install numpy scipy opencv pyrender open3d pybullet
```

## Running
1. Data generation and preprocessing
   ```
   python data_generate.py --dataset Sileance_Dataset --model_name gear --min_inst_num 1 --max_inst_num 60 --num_cycles_train 100 --num_cycles_val 10
   python data_preprocess.py --dataset Sileance_Dataset --model_name gear
   ```
3. Training
   ```
   # Config the file PointGroup/config/config_pointgroup.yaml then run the following command:
   python trainer_pointgroup.py
   # Config the file PoseEstimation/config/config_pose.yaml then run the following command:
   python trainer_gscpose.py
   ```
5. Evaluating
   ```
   python predict.py
   ```
