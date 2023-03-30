#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2023/03/29 4:05 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : pandas_utils.py
# @Software  : PyCharm

import pandas as pd
from sklearn.model_selection import ParameterGrid
from ..utils import read_yaml, get_region_grid

def stringify_list(df, col):
    df[col] = [','.join(map(str, l)) for l in df[col]]
    return df


def merge_dicts(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def get_all_settings(region_code='R4_64'):
    obj = {
        'debug_mode': [False],
        'predict_region': ['R4_64'],
        'horizon': [3, 4, 5, 6],
        'pm_type': ['PM10', 'PM25']
    }
    obj_list = list(ParameterGrid(obj))

    inner_grid = dict(
        is_reg=[True, False],
        model_name=['RNN', 'CNN'],
        model_type=['single', 'double']
    )
    inner_grids = list(ParameterGrid(inner_grid))

    settings = read_yaml('./data_folder/settings.yaml')
    param_list = []

    for param in obj_list:
        grids = get_region_grid(region_code, settings, param['pm_type'].lower())
        for grid in grids:
            for key in param.keys():
                grid[key] = param[key]
            grid['esv_years'] = settings['esv_years'][grid['periods']]
            for lag in [1, 2, 3, 4, 5, 6, 7]:
                grid['lag'] = lag
                for inner_grid in inner_grids:
                    merged = merge_dicts(inner_grid, grid)
                    merged[
                        'exp_name'] = f"{merged['predict_region']}/{merged['predict_region']}_{merged['representative_region']}_period_{merged['predict_region']}_rmgroup_{merged['remove_regions']}"
                    param_list.append(merged)

    return param_list
