#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn

class MLP_Dict(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layers=[128, 64, 32], activation_function="relu", output_keys=["T"]):
        """
        初始化 MLP_Dict。

        参数:
        - input_shape: 输入数据的形状。
        - output_shape: 输出数据的形状。
        - hidden_layers: 隐藏层的神经元数量列表。
        - activation_function：激活函数，relu/tanh
        - output_keys: 输出的键值列表
        """
        super(MLP_Dict, self).__init__()
        
        self.output_keys = output_keys  # 输出键列表
        
        # 输入层
        layers = []
        input_size = input_shape[0]  # 假设 input_shape 是 (input_dim,)
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            if activation_function == "relu":
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
        - 输出数据，格式为字典
        """
        output = self.network(x)
        
        # 将输出封装成字典形式
        outputs = {key: output for key in self.output_keys}  # 当前只有一个输出分量 "T"
        
        return outputs
