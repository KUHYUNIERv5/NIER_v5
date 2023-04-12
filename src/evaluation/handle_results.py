#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2023/04/05 1:25 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : handle_results.py
# @Software  : PyCharm

import pandas as pd
import numpy as np
import os
from src.utils import load_data, save_data, read_yaml, get_region_grid, merge_dicts, stringify_list
import uuid
from sklearn.model_selection import ParameterGrid
import datetime
import time
from tqdm import tqdm


def get_all_settings(region='R4_64'):
    obj = {
        'debug_mode': [False],
        'predict_region': [region],
        'horizon': [3, 4, 5, 6],
        'pm_type': ['PM10', 'PM25']
    }
    obj_list = list(ParameterGrid(obj))

    intermediate = dict(
        sampling=['normal', 'oversampling'],
        lag=[1, 2, 3, 4, 5, 6, 7]
    )
    intermediate_grid = list(ParameterGrid(intermediate))

    inner_grid = dict(
        is_reg=[True, False],
        model_name=['RNN', 'CNN'],
        model_type=['single', 'double']
    )
    inner_grids = list(ParameterGrid(inner_grid))

    settings = read_yaml('./data_folder/settings.yaml')
    param_list = []

    for param in obj_list:
        grids = get_region_grid(region, settings, param['pm_type'].lower())
        for grid in grids:
            for key in param.keys():
                grid[key] = param[key]
            grid['esv_years'] = settings['esv_years'][grid['periods']]

            for intermediate in intermediate_grid:
                for key in intermediate.keys():
                    grid[key] = intermediate[key]

                for inner_grid in inner_grids:
                    merged = merge_dicts(inner_grid, grid)
                    merged['run_type'] = 'regression' if merged['is_reg'] else 'classification'
                    del merged['is_reg']
                    del merged['debug_mode']
                    param_list.append(merged)

    all_settings = pd.DataFrame.from_records(param_list)
    all_settings.rename(columns={'periods': 'period_version', 'remove_regions': 'rm_region', 'model_name': 'model'},
                        inplace=True)
    return all_settings


def empty_result_df(length):
    c = np.array([2017, 2018, 2019, 2020, 2021])
    c = np.concatenate(np.tile(c, (10, 1)).T)
    vec = ['f1', 'accuracy', 'hit', 'pod', 'far']
    val = [f'val_{v}' for v in vec]
    test = [f'test_{v}' for v in vec]
    c2 = np.concatenate(np.tile(np.concatenate((val, test)), (5, 1)))

    col = pd.MultiIndex.from_arrays([c, c2])
    result_df = pd.DataFrame(np.full((length, 50), -1.), columns=col)

    return result_df


def get_region_result(exp_dir, region='R4_68'):
    root_dir = os.path.join(exp_dir, region)
    result_dir = os.path.join(root_dir, 'results')
    exp_settings = pd.read_csv(os.path.join(root_dir, 'id_list.csv'))
    print("id_list.csv loaded")
    ids = exp_settings['id'].tolist()
    ids = [str(id) for id in ids]

    result_list = [load_data(os.path.join(result_dir, f'{i}.pkl')) for i in tqdm(ids)]

    result_df = empty_result_df(len(result_list))

    for i, result in enumerate(tqdm(result_list)):
        for year in result['val_results']['best_results'].keys():
            for k in ['f1', 'accuracy', 'hit', 'pod', 'far']:
                result_df.loc[i, (year, f'val_{k}')] = result['val_results']['best_results'][year][k]
                result_df.loc[i, (year, f'test_{k}')] = result['test_results'][year]['test_result'][k]

    return_df = pd.concat([exp_settings, result_df], axis=1)
    return_df.to_excel(os.path.join(root_dir, f'{region}_result.xlsx'), engine='xlsxwriter')
    return return_df