#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2023/04/18 9:58 AM
# @Author    : Junhyung Kwon
# @Site      :
# @File      : run_v3.py
# @Software  : PyCharm

import sys

sys.path.append('../NIER_v5')

from sklearn.model_selection import ParameterGrid
from src.evaluation.v3_module import V3_Runner
from src.utils import save_data
import os
import argparse


def main(region, pm_type, save_dir, data_dir, root_dir, r4_dir, cmaq_dir):
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
    ),

    args_dict = dict(
        region=region,
        device='cpu',
        data_dir=data_dir,
        root_dir=root_dir,
        r4_dir=r4_dir,
        cmaq_dir=cmaq_dir,
        inference_type=2022,
        pm_type=pm_type,
        horizon=4,
        validation_days=7,
        model_num=100,
        add_r4_models=True,
        add_cmaq_model=True,
        model_args=model_args
    )

    horizons = [4, 5, 6]

    for horizon in horizons:
        args_dict['horizon'] = horizon
        v3_obj = V3_Runner(**args_dict)

        hyp_params = dict(
            mod=[0, 1],
            model_type_mod=[0, 1],
            num_top_k=[5, 15, 25]
        )

        obj_list = list(ParameterGrid(hyp_params))

        for obj in obj_list:
            mod = obj['mod']
            model_type_mod = obj['model_type_mod']
            num_top_k = obj['num_top_k']

            save_name = f'pm{pm_type}_horizon{horizon}_modeltype{model_type_mod}_f1limit{mod}_numtopk{num_top_k}.pkl'
            model_type_keys = ['cls_rnn', 'reg_rnn', 'reg_cnn'] if model_type_mod == 1 else None
            val_f1_limit = 1.1 if mod == 0 else 1.0

            v3_obj.initialize(model_type_keys, val_f1_limit)
            res = v3_obj.run_v3(top_k=num_top_k)
            save_data(res, save_dir, save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='v3 running')

    parser.add_argument('--debug', '-d', action="store_true")
    parser.add_argument('--pm_type', '-p', type=str, default="PM10")
    parser.add_argument('--region', '-r', help='input region', default='R4_68')
    parser.add_argument('--root_dir', '-rd', type=str, help='directory to save results',
                        default='/workspace/results/v5_phase2/')
    parser.add_argument('--data_dir', '-dd', type=str, default='/workspace/R5_phase2/')
    parser.add_argument('--r4_dir', '-rfd', type=str, default='/workspace/results/v3_um_r4to_r5')
    parser.add_argument('--cmaq_dir', '-c', type=str, default='/workspace/results/v3_cmaq/')

    args = parser.parse_args()

    data_dir = args.data_dir
    root_dir = args.root_dir
    r4_dir = args.r4_dir
    cmaq_dir = args.cmaq_dir
    pm_type = args.pm_type

    region = args.region
    save_dir = '/workspace/results/v3_r5/'
    save_dir = os.path.join(save_dir, region)
    os.makedirs(save_dir, exist_ok=True)

    main(region, pm_type, save_dir, data_dir, root_dir, r4_dir, cmaq_dir)
