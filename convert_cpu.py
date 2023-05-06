#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2023/04/05 3:40 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : convert_cpu.py
# @Software  : PyCharm

import os
from src.utils import load_data, save_data
from src.models import SingleInceptionModel_v2, DoubleInceptionModel_v2, SingleInceptionCRNN_v2, DoubleInceptionCRNN_v2
import pandas as pd
import argparse


def initialize_model(e, dropout=.1):
    pca_dim = dict(  # 건드리면 안됨
        obs=256,
        fnl=512,
        wrf=128,
        cmaq=512,
        numeric=512
    )

    model_args = dict(
        obs_dim=pca_dim['obs'],
        fnl_dim=pca_dim['fnl'],
        num_dim=pca_dim['numeric'],
        lag=e.lag
    )

    if e.horizon > 3:
        is_point_added = True
    else:
        is_point_added = False
    is_reg = True if e.run_type == 'regression' else False

    net = None

    if e.model == 'CNN':
        if e.model_type == 'single':
            net = SingleInceptionModel_v2(dropout=dropout, reg=is_reg, added_point=is_point_added,
                                          **model_args)
        elif e.model_type == 'double':
            net = DoubleInceptionModel_v2(dropout=dropout, reg=is_reg, added_point=is_point_added,
                                          **model_args)
        else:
            net = DoubleInceptionModel_v2(dropout=dropout, reg=is_reg, added_point=is_point_added,
                                          **model_args)
    elif e.model == 'RNN':
        if e.model_type == 'single':
            net = SingleInceptionCRNN_v2(dropout=dropout, reg=is_reg, rnn_type='GRU',
                                         added_point=is_point_added, **model_args)
        elif e.model_type == 'double':
            net = DoubleInceptionCRNN_v2(dropout=dropout, reg=is_reg, rnn_type='GRU',
                                         added_point=is_point_added, **model_args)
        else:
            net = DoubleInceptionCRNN_v2(dropout=dropout, reg=is_reg, rnn_type='GRU',
                                         added_point=is_point_added, **model_args)

    return net


def convert_model_save(ids, model_dir, exp_settings):
    for index, i in enumerate(ids):
        e = exp_settings.iloc[index]
        saved_models = load_data(os.path.join(model_dir, f'{i}.pkl'))
        net = initialize_model(e)
        net.load_state_dict(saved_models['model_weights'][2021])
        # net = saved_models['network'].cpu()
        model_weights = {
            'id': saved_models['id'],
            'network': net.cpu(),
            'model_weights': saved_models['model_weights']
        }
        if str(i) == str(saved_models['id']):
            save_data(model_weights, model_dir, f'{i}.pkl')
        else:
            print('ID not matched> original: ', i, " saved: ", saved_models['id'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='blabla')

    parser.add_argument('--dir', '-d', type=str)
    parser.add_argument('--region', '-r', type=str)

    args = parser.parse_args()
    root_dir = os.path.join(args.dir, args.region)

    # root_dir = '/workspace/results/v5_phase2/R4_68/'

    model_dir = os.path.join(root_dir, 'models')

    exp_settings = pd.read_csv(os.path.join(root_dir, 'id_list.csv'))
    ids = exp_settings['id'].tolist()
    ids = [str(id) for id in ids]

    convert_model_save(ids, model_dir, exp_settings)
