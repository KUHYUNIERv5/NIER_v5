import torch.nn as nn
import torch

from .transformer_modules import EncoderLayer, DecoderLayer, PositionalEncoding


class Transformer(nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, dec_seq_len, out_seq_len, n_decoder_layers=1,
                 n_encoder_layers=1, n_heads=1, align_type='scaled_dot'):
        """
        vanilla Transformer models
        :param dim_val: dimension of attention value sequence
        :param dim_attn: dimension of attention output sequence
        :param input_size: dimension of input
        :param dec_seq_len: decoder input sequence length
        :param out_seq_len: output sequence length
        :param n_decoder_layers: number of decoder layers
        :param n_encoder_layers: number of encoder layers
        :param n_heads: number of heads in multihead attention
        :param align_type: alignment type in attention block (scaled_dot, dot, generalized_kernel)
        """
        super(Transformer, self).__init__()

        # Initiate encoder and Decoder layers

        self.obs_encs = nn.ModuleList([
            EncoderLayer(dim_val, dim_attn, n_heads, align_type=align_type) for _ in range(n_encoder_layers)
        ])
        self.fnl_encs = nn.ModuleList([
            EncoderLayer(dim_val, dim_attn, n_heads, align_type=align_type) for _ in range(n_encoder_layers)
        ])

        self.decs = nn.ModuleList([
            DecoderLayer(dim_val, dim_attn, n_heads, align_type=align_type) for _ in range(n_decoder_layers)
        ])

        self.pos = PositionalEncoding(dim_val)

        # Dense layers for managing network inputs and outputs
        self.enc_input_fc1 = nn.Linear(input_size, dim_val)
        self.enc_input_fc2 = nn.Linear(input_size, dim_val)
        self.dec_input_fc = nn.Linear(input_size, dim_val)
        self.out_fc = nn.Linear(dec_seq_len * dim_val, out_seq_len)
        self.dec_seq_len = dec_seq_len

    def forward(self, x_obs, x_fnl, x_num=None):
        obs_e = self.pos(self.enc_input_fc1(x_obs))
        for enc in self.obs_encs:
            obs_e = enc(obs_e)

        fnl_e = self.pos(self.enc_input_fc2(x_fnl))
        for enc in self.fnl_encs:
            fnl_e = enc(fnl_e)

        e = torch.cat((x_obs, x_fnl), dim=1)
        e = e.view(e.shape[0], -1)

        d = self.pos(self.dec_input_fc(x_num))
        for dec in self.decs:
            d = dec(d, e)

        x = self.out_fc(d.flatten(start_dim=1))
        return x
