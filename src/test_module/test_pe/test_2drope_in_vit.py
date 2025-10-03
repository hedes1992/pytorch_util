#coding=utf-8
"""
implement 2D-ROPE in qwen2.5vl-VIT
"""

import torch
from torch import nn
import math
import unittest
import pdb

# 1. 生成逆频率向量
def precompute_inv_freq(model_dim, base=10000, dtype=torch.float32, device=None):
    """
    返回: [model_dim//2]的逆频率向量
    """
    inv_freq = 1.0 / (base ** torch.arange(0, model_dim, 2, device=device, dtype=dtype) / model_dim)
    # shape is [model_dim // 2]
    return inv_freq

# 2. rotate_half: 把后半部分的正负号翻转
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

# rope func
def apply_rotary_pos_emb(x, cos, sin):
    """
    x: [..., seq_len, model_dim]
    cos: [seq_len, model_dim]
    sin: [seq_len, model_dim]
    """
    return x * cos + rotate_half(x) * sin

# 2d-rope layer

class RoPE2D(nn.Module):
    def __init__(self, model_dim, base=10000):
        super().__init__()
        assert model_dim % 4 == 0, "model_dim must be divisible by 4"
        self.model_dim = model_dim
        self.base = base
        self.register_buffer("inv_freq", None, persistent=False)
    
    def _compute_cos_sin(self, h_pos, w_pos, device, dtype):
        """
        h_pos, w_pos [L]: patch的行列编号
        返回cos, sin: [L, model_dim]
        
        # 参考这里的5.2节
        """
        if self.inv_freq is None:
            self.inv_freq = precompute_inv_freq(self.model_dim, self.base, device=device, dtype=dtype)
        # 分别对h/w做 1d-rope(每半头分别编码)
        half = self.model_dim // 2
        # h 部分
        inv_freq_h = self.inv_freq[:half//2]   # 前 1/4 维度
        inv_freq_w = self.inv_freq[:half//2]   # 后 1/4 维度
        # 升维: [L, 1] @ [1, half//2] -> [L, half//2]
        angle_h = h_pos.float().unsqueeze(1) * inv_freq_h.unsqueeze(0)
        angle_w = w_pos.float().unsqueeze(1) * inv_freq_w.unsqueeze(0)
        # 拼接
        angle = torch.cat([angle_h, angle_w], dim=-1)  # [L, half]
        # 扩展到完整 dim(1d rope也是如此)
        angle = torch.cat([angle, angle], dim=-1)      # [L, dim]
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        return cos.to(dtype), sin.to(dtype)

    def forward(self, q, k, h_pos, w_pos):
        """
        q/k: [..., seq_len, model_dim]
        h_pos/w_pos: [seq_len]
        """
        # 和1d-rope最主要的区别在于这个compute_cos_sin的实现
        # 1d-rope只有pos输入
        # 2d-rope则需要h_pos和w_pos输入
        cos, sin = self._compute_cos_sin(h_pos, w_pos, q.device, q.dtype)
        # cos/sin 加维度和q/k一样
        q_shape_len = len(q.shape)
        cos_shape_len = len(cos.shape)
        for i in range(q_shape_len - cos_shape_len):
            # 扩充维度
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        q_out = apply_rotary_pos_emb(q, cos, sin)
        k_out = apply_rotary_pos_emb(k, cos, sin)
        return q_out, k_out

class TestRoPE2D(unittest.TestCase):
    def test_rope2d(self):
        h_num = 10
        w_num = 10
        model_dim = 8
        base = 10000
        rope2d = RoPE2D(model_dim, base)
        batch_size = 2
        seq_len = h_num * w_num
        q = torch.rand(batch_size, seq_len, model_dim)
        k = torch.rand(batch_size, seq_len, model_dim)

        # h_pos/w_pos
        # h_pos: 0,0,...,0,1,1,...,1,2,2,...,2,...,9,9,...,9
        # w_pos: 0,1,2,...,9,0,1,2,...,9,0,1,2,...,9,0,1,2,...,9
        h_pos = torch.arange(h_num).repeat_interleave(w_num)
        w_pos = torch.arange(w_num).repeat(h_num)
        rope2d = RoPE2D(model_dim, base)
        q_out, k_out = rope2d(q, k, h_pos, w_pos)
        print(f"q_shape: {q_out.shape}")
        print(f"k_shape: {k_out.shape}")

if __name__ == '__main__':
    unittest.main()