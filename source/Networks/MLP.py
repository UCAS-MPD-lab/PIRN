#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layers=[128, 64, 32], activation_function="relu"):
        """
        初始化 MLP。

        参数:
        - input_shape: 输入数据的形状。
        - output_shape: 输出数据的形状。
        - hidden_layers: 隐藏层的神经元数量列表。
        - activation_function：激活函数，relu/tanh
        """
        super(MLP, self).__init__()
        
        # 输入层
        layers = []
        input_size = input_shape[0]  # 假设 input_shape 是 (input_dim,)
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            if activation_function=="relu":
                layers.append(nn.ReLU())
            elif activation_function == "tanh":
                layers.append(nn.Tanh())
            input_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(input_size, output_shape[0]))  # 假设 output_shape 是 (output_dim,)
        
        # 构建网络
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播。

        参数:
        - x: 输入数据。

        返回:
        - 输出数据。
        """
        return self.network(x)

class MLP2(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layers=[128, 64, 32], activation_function="relu"):
        """
        初始化 MLP。

        参数:
        - input_shape: 输入数据的形状。
        - output_shape: 输出数据的形状。
        - hidden_layers: 隐藏层的神经元数量列表。
        - activation_function：激活函数，relu/tanh
        """
        super(MLP2, self).__init__()
        
        # 输入层
        layers = []
        input_size = input_shape[0]  # 假设 input_shape 是 (input_dim,)
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            if activation_function=="relu":
                layers.append(nn.ReLU())
            elif activation_function == "tanh":
                layers.append(nn.Tanh())
            input_size = hidden_size
        
        # 输出层：三个输出分别为 u, v, p
        self.fc_u = nn.Linear(input_size, 1)  # u 的输出
        self.fc_v = nn.Linear(input_size, 1)  # v 的输出
        self.fc_p = nn.Linear(input_size, 1)  # p 的输出
        
        # 构建网络
        self.network = nn.Sequential(*layers)

    def forward(self, x, y):
        """
        前向传播。

        参数:
        - x: 输入数据。

        返回:
        - 输出数据。
        """
        inputs = torch.cat((x.unsqueeze(1), y.unsqueeze(1)), dim=1)  # 合并 x 和 y 为 (batch_size, 2)
        # 通过网络进行前向传播
        hidden = self.network(inputs)

        # 分别计算 u, v 和 p
        u = self.fc_u(hidden)
        v = self.fc_v(hidden)
        p = self.fc_p(hidden)

        return u, v, p