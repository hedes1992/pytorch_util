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


DEF_batch_size = 2
DEF_channel_num = 4
DEF_height = 8
DEF_width = 8
DEF_bchw_shape = (DEF_batch_size, DEF_channel_num, DEF_height, DEF_width)

def batch_norm_ownfunc(input_tensor, nn_bn_layer):
    """
    input_tensor: tensor with shape (batch_size, channel_num, height, width)
    nn_bn_layer: nn.BatchNorm2d layer
    just implement the forward function of nn.BatchNorm2d
    """
    pdb.set_trace()
    pass

class TestNormInCNN(unittest.TestCase):
    def build_bchw_tensor(self, shape=DEF_bchw_shape):
        # build a tensor with shape (batch_size, channel_num, height, width)
        return torch.rand(shape)
    
    def test_batch_norm(self,):
        nn_bn_layer = nn.BatchNorm2d(DEF_channel_num)
        input_tensor = self.build_bchw_tensor()
        official_output_tensor = nn_bn_layer(input_tensor)
        own_output_tensor = batch_norm_ownfunc(input_tensor, nn_bn_layer)
        self.assertTrue(torch.allclose(official_output_tensor, own_output_tensor))
        
if __name__ == '__main__':
    unittest.main()