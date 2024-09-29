import torch.nn as nn
import torch
import torch.nn.functional as F

# from ..util.config import cfg


# class Rot_green(nn.Module):
#     def __init__(self):
#         super(Rot_green, self).__init__()
#         # self.f = cfg.feat_c_R
#         # self.k = cfg.R_c
#
#         self.f = 960
#         self.k = 3
#
#         self.conv1 = torch.nn.Conv1d(self.f, 1024, 1)
#         self.conv2 = torch.nn.Conv1d(1024, 256, 1)
#         self.conv3 = torch.nn.Conv1d(256, 256, 1)
#         self.conv4 = torch.nn.Conv1d(256, self.k, 1)
#         self.drop1 = nn.Dropout(0.2)
#         self.bn1 = nn.BatchNorm1d(1024)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.bn3 = nn.BatchNorm1d(256)
#
#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#
#         x = torch.max(x, 2, keepdim=True)[0]
#
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.drop1(x)
#         x = self.conv4(x)
#
#         x = x.squeeze(2)
#         x = x.contiguous()
#
#         return x


class Rot_green(nn.Module):
    def __init__(self):
        super(Rot_green, self).__init__()
        # self.f = cfg.feat_c_R
        # self.k = cfg.R_c

        self.f = 960
        self.k = 3

        self.conv1 = torch.nn.Conv1d(self.f, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x


# class Rot_red(nn.Module):
#     def __init__(self):
#         super(Rot_red, self).__init__()
#         # self.f = cfg.feat_c_R
#         # self.k = cfg.R_c
#
#         self.f = 960
#         self.k = 3
#
#         self.conv1 = torch.nn.Conv1d(self.f, 1024, 1)
#         self.conv2 = torch.nn.Conv1d(1024, 256, 1)
#         self.conv3 = torch.nn.Conv1d(256, 256, 1)
#         self.conv4 = torch.nn.Conv1d(256, self.k, 1)
#         self.drop1 = nn.Dropout(0.2)
#         self.bn1 = nn.BatchNorm1d(1024)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.bn3 = nn.BatchNorm1d(256)
#
#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#
#         x = torch.max(x, 2, keepdim=True)[0]
#
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.drop1(x)
#         x = self.conv4(x)
#
#         x = x.squeeze(2)
#         x = x.contiguous()
#
#         return x


class Rot_red(nn.Module):
    def __init__(self):
        super(Rot_red, self).__init__()
        # self.f = cfg.feat_c_R
        # self.k = cfg.R_c

        self.f = 960
        self.k = 3

        self.conv1 = torch.nn.Conv1d(self.f, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x