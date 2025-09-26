#coding=utf-8
"""
unittest for norm in CNN
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import _init_paths
import pdb


DEF_batch_size = 1
DEF_channel_num = 2
DEF_height = 4
DEF_width = 4
DEF_bchw_shape = (DEF_batch_size, DEF_channel_num, DEF_height, DEF_width)

def batch_norm_ownfunc(input_tensor, nn_bn_layer):
    """
    input_tensor: tensor with shape (batch_size, channel_num, height, width)
    nn_bn_layer: nn.BatchNorm2d layer
    just implement the forward function of nn.BatchNorm2d

    nn_bn_layer中的主要属性:
        在 nn.BatchNorm2d 里：
        可学习参数
        γ (gamma) → weight  (形状 = num_features)
        β (beta)  → bias   (形状 = num_features)
        初始化默认 γ=1，β=0，可通过 affine=False 关闭。
        运行统计量（不参与梯度更新，用于推理）
        running_mean  → 训练阶段每个 batch 均值的滑动平均
        running_var   → 训练阶段每个 batch 方差的滑动平均
    """
    mean = input_tensor.mean(dim=(0, 2, 3), keepdim=True)
    var = input_tensor.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
    output_tensor = (input_tensor - mean) / (torch.sqrt(var + nn_bn_layer.eps)) * nn_bn_layer.weight.view(1, -1, 1, 1) + nn_bn_layer.bias.view(1, -1, 1, 1)
    return output_tensor

class TestNormInCNN(unittest.TestCase):
    def build_bchw_tensor(self, shape=DEF_bchw_shape):
        # build a tensor with shape (batch_size, channel_num, height, width)
        return torch.rand(shape)
    
    def test_batch_norm(self,):
        nn_bn_layer = nn.BatchNorm2d(DEF_channel_num)
        nn_bn_layer.train()
        # eval() 模式的话会采用running_mean 和 running_var 进行归一化，而不是input_tensor的均值和方差
        input_tensor = self.build_bchw_tensor()
        official_output_tensor = nn_bn_layer(input_tensor)
        own_output_tensor = batch_norm_ownfunc(input_tensor, nn_bn_layer)
        self.assertTrue(torch.allclose(official_output_tensor, own_output_tensor))
        
if __name__ == '__main__':
    unittest.main()