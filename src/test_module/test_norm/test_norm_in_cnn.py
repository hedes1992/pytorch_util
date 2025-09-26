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
        γ (gamma) → weight  (形状 = num_features, 即 (1, c, 1, 1))
        β (beta)  → bias   (形状 = num_features, 即 (1, c, 1, 1))
        初始化默认 γ=1，β=0，可通过 affine=False 关闭。
        运行统计量（不参与梯度更新，用于推理）
        running_mean  → 训练阶段每个 batch 均值的滑动平均(默认有初始值)
        running_var   → 训练阶段每个 batch 方差的滑动平均(默认有初始值)
    """
    mean = input_tensor.mean(dim=(0, 2, 3), keepdim=True)
    var = input_tensor.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
    output_tensor = (input_tensor - mean) / (torch.sqrt(var + nn_bn_layer.eps)) * nn_bn_layer.weight.view(1, -1, 1, 1) + nn_bn_layer.bias.view(1, -1, 1, 1)
    return output_tensor

def layer_norm_ownfunc(input_tensor, nn_ln_layer):
    """
    input_tensor: tensor with shape (batch_size, channel_num, height, width)
    nn_ln_layer: nn.LayerNorm layer
        weight shape: (1, c, h, w), 看着显存占用较高
        没有bias(靠elementwise_affine参数控制是否有bias)
        且没有running_mean和running_var
    just implement the forward function of nn.LayerNorm

    """
    mean = input_tensor.mean(dim=(1, 2, 3), keepdim=True)
    var = input_tensor.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
    output_tensor = (input_tensor - mean) / (torch.sqrt(var + nn_ln_layer.eps)) * nn_ln_layer.weight.view(1, *nn_ln_layer.weight.shape) + nn_ln_layer.bias.view(1, *nn_ln_layer.bias.shape)
    return output_tensor

def instance_norm_ownfunc(input_tensor, nn_in_layer):
    """
    input_tensor: tensor with shape (batch_size, channel_num, height, width)
    nn_in_layer: nn.InstanceNorm2d layer
    有running_mean和running_var(初始值为None), weight(初始值为None), 但没有bias(靠affine=True控制是否有bias)
    just implement the forward function of nn.InstanceNorm2d
    """
    # 和layernorm不同的是, 是对每个样本的每个channel内部算mean和var
    # 但是gamma和beta参数的shape是(1, c, 1, 1), 较省显存
    mean = input_tensor.mean(dim=(2, 3), keepdim=True)
    var = input_tensor.var(dim=(2, 3), keepdim=True, unbiased=False)
    output_tensor = (input_tensor - mean) / (torch.sqrt(var + nn_in_layer.eps)) * nn_in_layer.weight.view(1, -1, 1, 1) + nn_in_layer.bias.view(1, -1, 1, 1)
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
    
    def test_layer_norm(self,):
        nn_ln_layer = nn.LayerNorm(DEF_bchw_shape[1:], elementwise_affine=True)
        nn_ln_layer.train()
        input_tensor = self.build_bchw_tensor()
        official_output_tensor = nn_ln_layer(input_tensor)
        own_output_tensor = layer_norm_ownfunc(input_tensor, nn_ln_layer)
        self.assertTrue(torch.allclose(official_output_tensor, own_output_tensor))
    
    def test_instance_norm(self,):
        nn_in_layer = nn.InstanceNorm2d(DEF_channel_num, track_running_stats=False, affine=True)
        nn_in_layer.train()
        input_tensor = self.build_bchw_tensor()
        official_output_tensor = nn_in_layer(input_tensor)
        own_output_tensor = instance_norm_ownfunc(input_tensor, nn_in_layer)
        self.assertTrue(torch.allclose(official_output_tensor, own_output_tensor))
        
if __name__ == '__main__':
    unittest.main()