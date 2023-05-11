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


def main(region, pm_type, save_dir, data_dir, root_dir, r4_dir, cmaq_dir, horizons=None, hyp_params=None,
         debug=False, equality=0):
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
        add_cmaq_model=True
    )

    pm10_all = [59, 60, 61, 62, 63, 67, 68, 71, 73, 74, 75, 77]
    pm10_all = [f'R4_{i}' for i in pm10_all]
    pm25_all = [59, 60, 61, 62, 63, 65, 66, 67, 68, 69, 72, 73, 76]
    pm25_all = [f'R4_{i}' for i in pm25_all]

    if hyp_params is None:
        hyp_params = dict(
            mod=[0],
            model_type_mod=[0, 1],
            num_top_k=[5, 15, 25, 35]
        )

    if horizons is None:
        horizons = [4, 5, 6]

    print('parameters and horizons: ', hyp_params, horizons)

    equality_on = True if equality == 1 else False

    if debug:
        horizon = horizons[0]
        mod = 0
        model_type_mod = 0
        num_top_k = 5

        if horizon == 3:
            if (pm_type == 'PM10' and region in pm10_all) or (pm_type == 'PM25' and region in pm25_all):
                h3_root_dir = os.path.join(root_dir, 'horizon_3_all')
            else:
                h3_root_dir = os.path.join(root_dir, 'horizon_3')
            args_dict['root_dir'] = h3_root_dir
        else:
            args_dict['root_dir'] = root_dir

        args_dict['horizon'] = horizon
        print(args_dict)
        v3_obj = V3_Runner(**args_dict)

        if equality_on:
            save_name = f'pm{pm_type}_horizon{horizon}_modeltype{model_type_mod}_f1limit{mod}_numtopk{num_top_k}_eq.pkl'
        else:
            save_name = f'pm{pm_type}_horizon{horizon}_modeltype{model_type_mod}_f1limit{mod}_numtopk{num_top_k}.pkl'
        print(save_name)
        model_type_keys = ['cls_rnn', 'reg_rnn', 'reg_cnn'] if model_type_mod == 1 else None
        val_f1_limit = 1.1 if mod == 0 else 1.0

        v3_obj.initialize(model_type_keys, val_f1_limit)
        res = v3_obj.run_v3(top_k=num_top_k, model_type_keys=model_type_keys, equality_on=equality_on, debug=debug)
        save_data(res, save_dir, save_name)

    else:
        for horizon in horizons:
            if horizon == 3:
                if (pm_type == 'PM10' and region in pm10_all) or (pm_type == 'PM25' and region in pm25_all):
                    h3_root_dir = os.path.join(root_dir, 'horizon_3_all')
                else:
                    h3_root_dir = os.path.join(root_dir, 'horizon_3')
                args_dict['root_dir'] = h3_root_dir
            else:
                args_dict['root_dir'] = root_dir

            args_dict['horizon'] = horizon
            v3_obj = V3_Runner(**args_dict)

            obj_list = list(ParameterGrid(hyp_params))

            for obj in obj_list:
                mod = obj['mod']
                model_type_mod = obj['model_type_mod']
                num_top_k = obj['num_top_k']
                if equality_on:
                    save_name = f'pm{pm_type}_horizon{horizon}_modeltype{model_type_mod}_f1limit{mod}_numtopk{num_top_k}_eq.pkl'
                else:
                    save_name = f'pm{pm_type}_horizon{horizon}_modeltype{model_type_mod}_f1limit{mod}_numtopk{num_top_k}.pkl'
                print(save_name)

                model_type_keys = ['cls_rnn', 'reg_rnn', 'reg_cnn'] if model_type_mod == 1 else None
                val_f1_limit = 1.1 if mod == 0 else 1.0

                v3_obj.initialize(model_type_keys, val_f1_limit)
                res = v3_obj.run_v3(top_k=num_top_k, model_type_keys=model_type_keys, equality_on=equality_on)
                # print(save_name)
                save_data(res, save_dir, save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='v3 running')

    parser.add_argument('--debug', '-d', action="store_true")
    parser.add_argument('--pm_type', '-p', type=str, default="PM10")
    parser.add_argument('--region', '-r', help='input region', default='R4_68')
    parser.add_argument('--equality', '-e', type=int, default=0)
    parser.add_argument('--horizons', '-ho', type=int, nargs='+', default=[4, 5, 6])
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
    save_dir = '/workspace/results/v3_r5_2/'
    save_dir = os.path.join(save_dir, region)
    os.makedirs(save_dir, exist_ok=True)

    hyp_params = None

    if args.equality == 1:
        hyp_params = dict(
            mod=[0],
            model_type_mod=[0, 1],
            num_top_k=[5, 15, 25, 35],
        )
    print(args.equality, hyp_params, args.horizons)

    main(region, pm_type, save_dir, data_dir, root_dir, r4_dir, cmaq_dir, horizons=args.horizons, hyp_params=hyp_params,
         debug=args.debug, equality=args.equality)
