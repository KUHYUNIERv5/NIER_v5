#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2023/01/05 9:35 AM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : model.py
# @Software  : PyCharm

import torch
from torch import nn, einsum
from einops.layers.torch import Rearrange
from einops import rearrange

def alignment_function(Q, K, type):
    m = None
    if type == 'scaled_dot':
        m = torch.matmul(Q, K.transpose(2, 1).float())
        m /= torch.sqrt(torch.tensor(Q.shape[-1]).float())
    elif type == 'dot':
        m = torch.matmul(Q, K.transpose(2, 1).float())

    return torch.softmax(m, -1)

def attention(Q, K, V, align_type):
    # Attention(Q, K, V) = norm(QK)V
    a = alignment_function(Q, K, align_type)  # (batch_size, dim_attn, seq_length)

    return torch.matmul(a, V)  # (batch_size, seq_length, seq_length)


class AttentionWeight(nn.Module):
    def __init__(self, dim_input, dim_val):
        super(AttentionWeight, self).__init__()
        self.dim_val = dim_val
        self.fc1 = nn.Linear(dim_input, dim_val, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, dim_val, dim_attn, align_type):
        super(AttentionBlock, self).__init__()
        self.value = AttentionWeight(dim_val, dim_val)
        self.key = AttentionWeight(dim_val, dim_attn)
        self.query = AttentionWeight(dim_val, dim_attn)
        self.align_type = align_type

    def forward(self, x, kv=None):
        if (kv is None):
            # Attention with x connected to Q,K and V (For encoder)
            return attention(self.query(x), self.key(x), self.value(x), self.align_type)

        # Attention with x as Q, external vector kv as K an V (For decoder)
        return attention(self.query(x), self.key(kv), self.value(kv), self.align_type)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads, align_type='scaled_dot'):
        super(MultiHeadAttentionBlock, self).__init__()
        assert align_type in ['scaled_dot', 'dot', 'generalized_kernel']
        self.heads = []
        for i in range(n_heads):
            self.heads.append(AttentionBlock(dim_val, dim_attn, align_type))

        self.heads = nn.ModuleList(self.heads)

        self.fc = nn.Linear(n_heads * dim_val, dim_val, bias=False)

    def forward(self, x, kv=None):
        a = []
        for h in self.heads:
            a.append(h(x, kv=kv))

        a = torch.stack(a, dim=-1)  # combine heads
        a = a.flatten(start_dim=2)  # flatten all head outputs

        x = self.fc(a)

        return x



class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super(Attention, self).__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, self.hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(self.hidden_dim, dim, 1)

        self.rearrange_layer1 = Rearrange('b (h c) x -> b h c x', h=self.heads)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: self.rearrange_layer1(t), qkv
        )
        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)