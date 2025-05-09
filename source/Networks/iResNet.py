#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class iResNet1D(nn.Module):
    def __init__(self, input_shape, full_connect_shape, q, N, output_keys):
        super(iResNet1D, self).__init__()
        self.input_shape = input_shape
        self.full_connect_shape = full_connect_shape  # 格式：(Channel, H, W)
        self.q = q
        self.N = N
        self.output_keys = output_keys

        # 从 full_connect_shape 提取通道数 C
        self.C, self.H, self.W = full_connect_shape

        # 全连接层：将输入扩展到 (C * H * W)
        self.fc = nn.Linear(np.prod(input_shape), np.prod(full_connect_shape))

        # 创建 q 个 BB
        self.bb_layers = nn.ModuleList()
        for _ in range(q):
            sb_layers = nn.ModuleList()
            for _ in range(N):
                # 动态设置 in_channels 和 out_channels 为 C
                sb_layers.append(nn.Conv2d(self.C, self.C, kernel_size=(1,3), stride=1, padding=(0,1)))
                sb_layers.append(nn.LeakyReLU(0.01))
            self.bb_layers.append(sb_layers)

        # 最后一层：输出通道数为 1
        self.final_conv = nn.Conv2d(self.C, 1, kernel_size=(1,3), stride=1, padding=(0,1))


    def forward(self, x):
        # 全连接层 + Reshape
        x = self.fc(x)
        x = x.view(-1, self.C, self.H, self.W)  # 调整为 (batch_size, C, H, W)

        # 处理每个 BB
        for sb_layers in self.bb_layers:
            bb_input = x
            for layer in sb_layers:
                x = layer(x)
            x = (bb_input + x) * 2

        # 最后一层
        T = self.final_conv(x)

        # 将输出改为字典形式
        outputs = {
            key: T for key in self.output_keys # 当前只有一个输出分量，键为 "T"
        }
        return outputs


class iResNet2D(nn.Module):
    def __init__(self, input_shape, full_connect_shape, q, N, output_keys):
        super(iResNet2D, self).__init__()
        self.input_shape = input_shape
        self.full_connect_shape = full_connect_shape  # 格式：(Channel, H, W)
        self.q = q
        self.N = N
        self.output_keys = output_keys

        # 从 full_connect_shape 提取通道数 C
        self.C, self.H, self.W = full_connect_shape

        # 全连接层：将输入扩展到 (C * H * W)
        self.fc = nn.Linear(np.prod(input_shape), np.prod(full_connect_shape))

        # 创建 q 个 BB
        self.bb_layers = nn.ModuleList()
        for _ in range(q):
            sb_layers = nn.ModuleList()
            for _ in range(N):
                # 动态设置 in_channels 和 out_channels 为 C
                sb_layers.append(nn.Conv2d(self.C, self.C, kernel_size=(3, 3), stride=1, padding=(1, 1)))  # 修改为 (3, 3)
                sb_layers.append(nn.LeakyReLU(0.01))
            self.bb_layers.append(sb_layers)

         # 为每个输出键生成不同的卷积层
        self.final_convs = nn.ModuleDict()
        for key in self.output_keys:
            self.final_convs[key] = nn.Conv2d(self.C, 1, kernel_size=(3, 3), stride=1, padding=(1, 1))  # 修改为 (3, 3)

    def forward(self, x):
        # 全连接层 + Reshape
        x = self.fc(x)
        x = x.view(-1, self.C, self.H, self.W)  # 调整为 (batch_size, C, H, W)

        # 处理每个 BB
        for sb_layers in self.bb_layers:
            bb_input = x
            for layer in sb_layers:
                x = layer(x)
            x = (bb_input + x) * 2

        # 为每个输出键生成不同的张量
        outputs = {}
        for key in self.output_keys:
            outputs[key] = self.final_convs[key](x)
            
        return outputs


class iResNet3D(nn.Module):
    def __init__(self, input_shape, full_connect_shape, q, N, output_keys):
        super(iResNet3D, self).__init__()
        self.input_shape = input_shape
        self.full_connect_shape = full_connect_shape  # 格式：(Channel, D, H, W)
        self.q = q
        self.N = N
        self.output_keys = output_keys  # 需要输出的键

        # 从 full_connect_shape 提取通道数 C 和深度 D
        self.C, self.D, self.H, self.W = full_connect_shape

        # 全连接层：将输入扩展到 (C * D * H * W)
        self.fc = nn.Linear(np.prod(input_shape), np.prod(full_connect_shape))

        # 创建 q 个 BB
        self.bb_layers = nn.ModuleList()
        for _ in range(q):
            sb_layers = nn.ModuleList()
            for _ in range(N):
                # 动态设置 in_channels 和 out_channels 为 C
                sb_layers.append(nn.Conv3d(self.C, self.C, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)))  # 修改为 (3, 3, 3)
                sb_layers.append(nn.LeakyReLU(0.01))
            self.bb_layers.append(sb_layers)

        # 为每个输出键生成不同的卷积层
        self.final_convs = nn.ModuleDict()
        for key in self.output_keys:
            self.final_convs[key] = nn.Conv3d(self.C, 1, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))

    def forward(self, x):
        # 全连接层 + Reshape
        x = self.fc(x)
        x = x.view(-1, self.C, self.D, self.H, self.W)  # 调整为 (batch_size, C, D, H, W)

        # 处理每个 BB
        for sb_layers in self.bb_layers:
            bb_input = x
            for layer in sb_layers:
                x = layer(x)
            x = (bb_input + x) * 2

        # 为每个输出键生成不同的张量
        outputs = {}
        for key in self.output_keys:
            outputs[key] = self.final_convs[key](x)

        return outputs


