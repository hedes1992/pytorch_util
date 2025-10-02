#coding=utf-8
"""
Test the RoPE module.
"""

import torch
from torch import nn, Tensor
import math
import unittest
import pdb

"""
roformer style: 
    1. 相邻两维分别进行cos和sin变换，然后合并最后两维
    2. cos/sin 需要存储半个大小来用到apply_rope_roformer_style
    3. 不方便向量化
llama style: 
    1. 先旋转一半，再旋转另一半，最后合并最后两维
    2. cos/sin 需要存储整个来用到apply_rope_llama_style
    3. 向量化方便

二者在q_rot和k_rot内部的维度上不一致(model_dim维度上不一致)
但是对位置m和n的旋转的内积
inner_product(q_rot[m], k_rot[n]), 二者的结果是等价的

具体原因如下:
用复数视角一句话说明
RoPE 的本质是给每对维度 (x, y) 乘上复数相位 e^{iθ}。
GPT-J：把向量看成
z₀ = x₀ + i x₁, z₁ = x₂ + i x₃, …
GPT-NeoX：把向量看成
w₀ = x₀ + i x_{d/2}, w₁ = x₁ + i x_{d/2+1}, …
这两组复数 {z_k} 与 {w_k} 之间只差一个固定的置换矩阵 P（每行每列只有一个 1 其余 0）。
对任意位置 p 的 q、k 做旋转后，再算内积：
⟨RoPE(q,p), RoPE(k,p)⟩
= Re(∑_k z_k^q · conj(z_k^k))
= Re(∑_k w_k^q · conj(w_k^k))  （因为只是重排求和顺序）
所以数值结果完全一样，不是“近似一致”，也不是“只要大家都用相同方式就行”，而是数学上恒等

参考: https://www.kimi.com/share/d3fauuh3ntnobpb449gg
"""

def apply_rope_roformer_style(x: Tensor, cos: Tensor, sin: Tensor):
    """
    x: [..., seq, head_dim]
    cos/sin: [seq, head_dim]

    rope: roformer style or GPT-J style
    """
    # 相邻两维看成复数(x0, x1) -> x0 + i x1
    # 1. 取奇数偶数坐标
    x1, x2 = x[..., ::2], x[..., 1::2]
    # shape = [..., seq, head_dim//2, 2]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    # 合并最后两维 -> [..., seq, head_dim]
    return rotated.flatten(-2)

def apply_rope_llama_style(x: Tensor, cos: Tensor, sin: Tensor):
    """
    huggingface / llama / gpt-neox style
    """
    def rotate_half(x: Tensor):
        model_dim = x.shape[-1]
        x1 = x[..., :model_dim//2]
        x2 = x[..., model_dim//2:]
        return torch.cat([-x2, x1], dim=-1)
    x_rot = x * cos + rotate_half(x) * sin
    return x_rot

class RoPE(nn.Module):
    """
    旋转位置编码 (RoPE)
    只负责生成 cos/sin 缓存，真正“旋转”由 apply_rope 完成，
    方便在 Multi-Head Attention 里对 q、k 分别调用。
    """
    def __init__(self, model_dim: int, max_seq_len: int, base: int=10000, style='roformer'):
        super().__init__()
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)
        self.style = style
    
    def build_cache(self, seq_len: int):
        """
        提前计算得到的cos/sin缓存, 长度不够时再扩充
        """
        if self.cos is not None and self.cos.size(0) >= self.max_seq_len:
            # 有可用的缓存就直接返回
            return
        # 1. 计算theta_i = 10000 ^ (-2i/model_dim), i=0,1,2,...,model_dim//2-1
        theta = self.base ** (torch.arange(0, self.model_dim, 2) / self.model_dim)
        inv_freq = 1. / theta
        # 2. 位置 t = 0, 1, 2,..., seq_len-1
        t = torch.arange(seq_len, dtype=inv_freq.dtype, device=inv_freq.device)
        # 3. 外积得到 t * 1. / theta, shape=(seq_len, model_dim//2)
        freqs = torch.outer(t, inv_freq)
        if self.style == 'roformer':
            emb = freqs
        elif self.style == 'llama':
            # 4. 复制到 [seq_len, dim]（每两个维度共享同一个角度）
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            pdb.set_trace()
            pass
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)
        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len

    def forward(self, x: Tensor, seq_len=None):
        """
        x: [batch, seq, n_heads, head_dim]
        返回旋转后的张量，形状不变
        """
        if seq_len is None:
            seq_len = x.size(1)
        self.build_cache(seq_len)
        if self.style == 'roformer':
            return apply_rope_roformer_style(x, self.cos[:seq_len], self.sin[:seq_len])
        elif self.style == 'llama':
            return apply_rope_llama_style(x, self.cos[:seq_len], self.sin[:seq_len])
        else:
            pdb.set_trace()
            pass

class RoPETest(unittest.TestCase):
    def test_rope(self):
        batch_size, seq_len, n_heads, head_dim = 2, 10, 4, 8
        model_dim = n_heads * head_dim
        max_seq_len = seq_len
        rope = RoPE(model_dim=model_dim, max_seq_len=max_seq_len, style='roformer')
        rope = RoPE(model_dim=model_dim, max_seq_len=max_seq_len, style='llama')
        q = torch.randn(batch_size, seq_len, n_heads * head_dim)
        k = torch.randn(batch_size, seq_len, n_heads * head_dim)
        q_rot = rope(q)
        k_rot = rope(k)
        self.assertEqual(q_rot.shape, k_rot.shape)
        print(f"q_rot shape: {q_rot.shape}")
        print(f"k_rot shape: {k_rot.shape}")

if __name__ == '__main__':
    unittest.main()