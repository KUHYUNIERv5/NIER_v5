import numpy as np
import torch
import torch.nn as nn
import math

""" sinusoid position embedding """
def positional_encoding(n_seq, d_hidn):
    def cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)

    def get_posi_angle_vec(position):
        return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

    sinusoid_table = torch.tensor([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # even index sin
    sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # odd index cos

    return sinusoid_table.double()


# def a_norm(Q, K):
#     m = torch.matmul(Q, K.transpose(2, 1).float())
#     m /= torch.sqrt(torch.tensor(Q.shape[-1]).float())
#
#     return torch.softmax(m, -1)


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


class AttentionWeight(nn.Module):
    def __init__(self, dim_input, dim_val):
        super(AttentionWeight, self).__init__()
        self.dim_val = dim_val
        self.fc1 = nn.Linear(dim_input, dim_val, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        positional encoding module ref by https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        :param d_model: models dimension
        :param dropout: dropout probability (0~1)
        :param max_len: maximum length of sequence for positional encoding
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x


def get_data(batch_size, input_sequence_length, output_sequence_length):
    i = input_sequence_length + output_sequence_length

    t = torch.zeros(batch_size, 1).uniform_(0, 20 - i).int()
    b = torch.arange(-10, -10 + i).unsqueeze(0).repeat(batch_size, 1) + t

    s = torch.sigmoid(b.float())
    return s[:, :input_sequence_length].unsqueeze(-1), s[:, -output_sequence_length:]
