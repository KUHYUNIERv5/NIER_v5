#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2023/04/29 3:35 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : xai_dat.py
# @Software  : PyCharm

import os
import pandas as pd
import numpy as np
import datetime
from src.utils import load_data, save_data

def select_data_xai(region, pm_type, horizon, data_dir='/workspace/R5_phase2/', root_dir='/workspace/results/v5_phase2/'):
    root_dir = os.path.join(root_dir, region)
    csv_dir = os.path.join(root_dir, 'id_list.csv')
    pca_dim = dict(
        obs=256,
        fnl=512,
        wrf=128,
        cmaq=512,
        numeric=512
    )
    exp_settings = pd.read_csv(csv_dir)

    str_expr = f"(predict_region == '{region}') and (pm_type == '{pm_type}') and (horizon == {horizon})"
    exp = exp_settings.query(str_expr)

    e = exp.iloc[0]

    representative_region = e.representative_region
    period_version = e.period_version
    rm_region = e.rm_region

    file_name = os.path.join(region, f'{region}_{representative_region}_period_{period_version}_rmgroup_{rm_region}')
    data_file = os.path.join(data_dir, f'{file_name}.pkl')
    whole_data = load_data(data_file)


def handle_xai_data(whole_data, region):
    pca_dim = dict(
        obs=256,
        fnl=512,
        wrf=128,
        cmaq=512,
        numeric=512
    )

    y_PM10 = pd.concat([whole_data['obs']['PM10']['train_y'][region], whole_data['obs']['PM10']['test_y'][region]])
    y_PM25 = pd.concat([whole_data['obs']['PM25']['train_y'][region], whole_data['obs']['PM25']['test_y'][region]])
    mean_PM10, scale_PM10 = whole_data['obs']['PM10']['mean'], whole_data['obs']['PM10']['scale']
    mean_PM25, scale_PM25 = whole_data['obs']['PM25']['mean'], whole_data['obs']['PM25']['scale']

    obs_all = pd.concat([whole_data['obs']['X'][region][f"pca_{pca_dim['obs']}"]['train'],
                         whole_data['obs']['X'][region][f"pca_{pca_dim['obs']}"]['test']])
    obs_pca = whole_data['obs']['pca'][region]
    fnl_all = pd.concat([whole_data['fnl']['X'][f"pca_{pca_dim['fnl']}"]['train'],
                         whole_data['fnl']['X'][f"pca_{pca_dim['fnl']}"]['test']])
    fnl_pca = whole_data['fnl']['pca']
    num_all = pd.concat([whole_data['numeric']['X'][f"pca_{pca_dim['numeric']}"]['train'].reset_index(),
                         whole_data['numeric']['X'][f"pca_{pca_dim['numeric']}"]['test'].reset_index()])
    num_all = num_all.set_index(['RAW_DATE'])
    num_pca = whole_data['numeric']['pca']

    save_dicts = dict(
        obs=obs_all,
        fnl=fnl_all,
        num=num_all,
        obs_pca=obs_pca,
        fnl_pca=fnl_pca,
        num_pca=num_pca
    )

    save_dicts[f'PM10'] = {
        'y': y_PM10,
        'mean': mean_PM10,
        'scale': scale_PM10
    }
    save_dicts[f'PM25'] = {
        'y': y_PM25,
        'mean': mean_PM25,
        'scale': scale_PM25
    }

    return save_dicts

def make_xai_dataset(data_dir):
    regions = [f'R4_{i}' for i in np.arange(68, 78)]
    for region in regions:
        region_dir = os.path.join(data_dir, region)
        for file in os.listdir(region_dir):
            if not '_xai' in file and not '_v3' in file and file.endswith('.pkl'):
                name = file.split('.')[0] + '_xai'
                print(name)
                whole_data = load_data(os.path.join(region_dir, file))
                xai_data = handle_xai_data(whole_data, region)
                save_data(xai_data, region_dir, name + '.pkl')
