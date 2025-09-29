#coding=utf-8
"""
unit test for activation function in bchw format
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


DEF_batch_size = 1
DEF_channel_num = 2
DEF_height = 4
DEF_width = 4
DEF_bchw_shape = (DEF_batch_size, DEF_channel_num, DEF_height, DEF_width)
DEF_model_dim = 16

class SwiGLU(nn.Module):
    def __init__(self, model_dim, hidden_dim=None, bias=False):
        super(SwiGLU, self).__init__()
        if hidden_dim is None:
            # 保持参数量 ≈ 4×d_model
            hidden_dim = int(2 / 3 * 4 * model_dim)
            # 保证 hidden_dim 为 8 的倍数来进行加速
            hidden_dim = (hidden_dim + 7) // 8 * 8
            # 参数量是 3 * (model_dim * hidden_dim) ≈ 8*model_dim
        self.gate = nn.Linear(model_dim, hidden_dim, bias=bias)
        self.up = nn.Linear(model_dim, hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, model_dim, bias=bias)
    
    def forward(self, input_tensor):
        gate_output = self.gate(input_tensor)
        up_output = self.up(input_tensor)
        down_output = self.down(F.silu(gate_output) * up_output)
        return down_output

class TestActInBCHW(unittest.TestCase):
    def build_bchw_tensor(self, shape=DEF_bchw_shape):
        """
        build a tensor in bchw format
        """
        return torch.rand(shape)
    def build_shape_tensor(self, shape):
        """
        build a tensor in bc format
        """
        return torch.rand(shape)
    
    def test_swiglu(self):
        """
        test swiglu activation function
        """
        input_tensor = self.build_shape_tensor(shape=(DEF_batch_size, DEF_model_dim))
        swiglu = SwiGLU(model_dim=DEF_model_dim)
        swiglu_output = swiglu(input_tensor)
    
if __name__ == '__main__':
    unittest.main()