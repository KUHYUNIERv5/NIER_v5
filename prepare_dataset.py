#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2023/03/10 3:15 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : prepare_dataset.py
# @Software  : PyCharm

from src.dataset import MakeNIERDataset
from src.utils import load_data, read_yaml, all_equal, concatenate
from sklearn.model_selection import ParameterGrid
import numpy as np


def get_best_hyperparam(predict_region, settings):
    representatives = []
    for group in settings['region_groups']:
        if predict_region in settings['region_groups'][group]['regions']:
            representatives.append(settings['region_groups'][group]['representative'])

    res = dict()
    check_equality = lambda li: li if not all_equal(li) else [li[0]]
    for representative in representatives:
        res[representative] = dict(
            pm10=dict(
                periods=[],
                remove_regions=[],
                representative_region=[representative]
            ),
            pm25=dict(
                periods=[],
                remove_regions=[],
                representative_region=[representative]
            )
        )
        res[representative]['pm10']['periods'] = check_equality(
            settings['best_hyperparams']['period_optimized']['pm10'][representative])
        res[representative]['pm10']['remove_regions'] = check_equality(
            settings['best_hyperparams']['regions_optimized']['pm10'][representative])
        res[representative]['pm25']['periods'] = check_equality(
            settings['best_hyperparams']['period_optimized']['pm25'][representative])
        res[representative]['pm25']['remove_regions'] = check_equality(
            settings['best_hyperparams']['regions_optimized']['pm25'][representative])

    return res


def get_region_grid(region, settings):
    hyperparam = get_best_hyperparam(region, settings)

    final_grid = None
    for k in hyperparam.keys():
        hyperparam[k]
        pm10_grid = np.array(list(ParameterGrid(hyperparam[k]['pm10'])))
        pm25_grid = np.array(list(ParameterGrid(hyperparam[k]['pm25'])))

        final_grid = concatenate(final_grid, pm10_grid)
        final_grid = concatenate(final_grid, pm25_grid)

    res = list(map(dict, set(tuple(sorted(sub.items())) for sub in final_grid)))
    return res

if __name__ == '__main__':
    settings = read_yaml('./data_folder/settings.yaml')
    # region_lists = [f'R4_{i}' for i in range(59, 69)]
    region_lists = ['R4_62', ] #'R4_63', 'R4_65','R4_59', 'R4_60', 'R4_61',

    data_param = dict(
        reset_db=False,
        period_version='p3',
        test_period_version='v2',
        seed=999,
        preprocess_root='/workspace/data/NIERDataset/R5_phase2/data_folder',
        root_dir="/workspace/data/NIERDataset/R5_phase2/data_folder",
        save_processed_data=True,
        run_pca=True,
        predict_region='R4_62',
        representative_region='R4_62',
        remove_region=0,
        rmgroup_file='./data_folder/height_region_list.csv',
        yaml_dir='./static.yaml'
    )

    for region in region_lists:
        res = get_region_grid(region, settings)
        data_param['predict_region'] = region
        for grid in res:
            print(grid)
            data_param['period_version'] = grid['periods']
            data_param['remove_region'] = grid['remove_regions']
            data_param['representative_region'] = grid['representative_region']

            _ = MakeNIERDataset(**data_param)


