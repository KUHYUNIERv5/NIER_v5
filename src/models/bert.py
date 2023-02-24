#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/08/19 9:38 오후
# @Author    : Junhyung Kwon
# @Site      :
# @File      : BERT.py
# @Software  : PyCharm

import torch.nn as nn

from .transformer_modules import EncoderLayer, PositionalEncoding

class BERT(nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, enc_seq_len, out_seq_len, n_encoder_layers=1,
                 n_heads=1):
        """
        Vanilla BERT models with multihead attention layers
        :param dim_val: dimension of input value
        :param dim_attn: dimension of attention output
        :param input_size: input sequence length
        :param dec_seq_len: decoder input sequence length
        :param out_seq_len: output sequence length
        :param n_encoder_layers: number of encoder layers
        :param n_heads: number of heads in multihead attention
        """
        super(BERT, self).__init__()

        # Initiate encoder and Decoder layers
        encs = []
        for i in range(n_encoder_layers):
            encs.append(EncoderLayer(dim_val, dim_attn, n_heads))

        self.encs = nn.Sequential(*encs)

        self.pos = PositionalEncoding(dim_val)

        # Dense layers for managing network inputs and outputs
        self.enc_input_fc = nn.Linear(input_size, dim_val)
        self.enc_input_fc2 = nn.Linear(dim_val, dim_val)
        self.out_fc = nn.Linear(enc_seq_len * dim_val, out_seq_len)

    def forward(self, x):
        # encoder
        e = self.enc_input_fc(x)
        # pe = positional_encoding(e.size(1), e.size(2))
        e = self.pos(e)
        e = self.enc_input_fc2(e)
        e = self.encs(e)
        x = self.out_fc(e.flatten(start_dim=1))
        return x