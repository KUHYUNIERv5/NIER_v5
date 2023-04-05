#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2023/04/05 3:40 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : convert_cpu.py
# @Software  : PyCharm

import os
from src.utils import load_data, save_data
import pandas as pd
import argparse


def convert_model_save(ids, model_dir):
    for i in ids:
        saved_models = load_data(os.path.join(model_dir, f'{i}.pkl'))
        net = saved_models['network'].cpu()
        model_weights = {
                            'id': saved_models['id'],
                            'network': net,
                            'model_weights': saved_models['model_weights']
                        }
        if str(i) == str(saved_models['id']):
            save_data(model_weights, model_dir, f'{i}.pkl')
        else:
            print('ID not matched> original: ',i," saved: ", saved_models['id'])


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

    convert_model_save(ids, model_dir)
