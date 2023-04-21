#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2023/04/11 10:39 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : inference_module.py
# @Software  : PyCharm

import os
import numpy as np
import pandas as pd
from copy import copy, deepcopy
from src.utils import save_data, load_data
from src.dataset import NIERDataset
from src.trainer import BasicTrainer
import ast
import torch
import time
from tqdm.auto import tqdm


def inference_dataset(predict_region, pm_type, horizon, period_version, rm_region, exp_name,
                      sampling, lag, data_dir, pca_dim, numeric_type, numeric_data_handling='normal',
                      numeric_scenario=4, seed=999, co2_load=False, type=2021):
    dataset_args = dict(
        predict_location_id=predict_region,
        predict_pm=pm_type,
        shuffle=False,
        sampling=sampling,
        data_path=data_dir,
        data_type='train',
        pca_dim=pca_dim,
        lag=lag,
        numeric_type=numeric_type,
        numeric_data_handling=numeric_data_handling,
        horizon=horizon,
        max_lag=7,
        max_horizon=6,
        numeric_scenario=numeric_scenario,
        timepoint_day=4,
        interval=1,
        seed=seed,
        serial_y=False,
        period_version=period_version,
        test_period_version='v2',
        esv_year=2021,  # early stopping validation year
        co2_load=co2_load,
        rm_region=rm_region,
        exp_name=exp_name
    )

    if type == 2022:
        dataset_args['data_type'] = 'test'
    else:
        dataset_args['data_type'] = 'valid'
        dataset_args['period_version'] = 'p5'

    inference_set = NIERDataset(**dataset_args)

    if co2_load:
        obs_dim = dataset_args['pca_dim']['obs'] + 3
    else:
        obs_dim = dataset_args['pca_dim']['obs']

    return inference_set, obs_dim


def inference(net, test_set, pm_type, model_name, model_type, is_reg, optimizer_name='SGD',
              objective_name='CrossEntropyLoss', batch_size=256, n_epochs=1, dropout=0., device='cpu', seed=999):
    trainer_args = dict(
        pm_type=pm_type,
        model_name=model_name,
        model_type=model_type,
        model_ver='v2',
        scheduler="MultiStepLR",
        lr_milestones=[40],
        optimizer_name=optimizer_name,
        objective_name=objective_name,
        n_epochs=n_epochs,
        dropout=dropout,
        device=device,
        name="NIER_R5_phase2",
        is_reg=is_reg,
        log_path="../log",
        seed=seed,
        log_flag=False,
    )

    optimizer_args = {
        'lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 1e-5,
        'nesterov': True
    }
    objective_args = {}

    trainer = BasicTrainer(**trainer_args)

    test_score, test_orig_pred, test_pred_score, test_label_score = trainer.test(test_set, net, batch_size,
                                                                                 optimizer_args, {})

    return test_score, test_orig_pred, test_pred_score, test_label_score


def inference_on_validset(region='R4_68', device='cpu', data_dir='/workspace/R5_phase2/',
                      root_dir='/workspace/results/v5_phase2/', inference_type=2021, debug=False):
    print(f'region: {region} | inference type: {inference_type}')
    root_dir = os.path.join(root_dir, region)

    model_dir = os.path.join(root_dir, 'models')
    tmp_dir = os.path.join(root_dir, 'tmp')

    csv_dir = os.path.join(root_dir, 'id_list.csv')
    exp_settings = pd.read_csv(csv_dir)

    exp_settings.rename(columns={'esv_years': 'esv_year'}, inplace=True)

    for k in ['f1', 'accuracy', 'hit', 'pod', 'far']:
        exp_settings[f'val_{k}'] = -1.

    tmp_df = []

    for i, exp_setting in tqdm(enumerate(exp_settings.iterrows())):
        now = time.time()
        series_list = []
        e = exp_setting[1]
        esv_year = ast.literal_eval(e.esv_year)
        file_name = f'{e.id}.pkl'
        model_data = load_data(os.path.join(model_dir, file_name))
        net = model_data['network']

        # set params
        predict_region = region
        representative_region = e.representative_region
        pm_type = e.pm_type
        sampling = e.sampling
        horizon = e.horizon
        lag = e.lag
        rm_region = e.rm_region
        exp_name = os.path.join(predict_region,
                                f'{predict_region}_{representative_region}_period_{e.period_version}_rmgroup_{rm_region}')

        model_name = e.model
        model_type = e.model_type
        is_reg = True if e.run_type == 'regression' else False

        period_version = 'p5'  # 2021년 데이터
        pca_dim = dict(  # 건드리면 안됨
            obs=256,
            fnl=512,
            wrf=128,
            cmaq=512,
            numeric=512
        )
        numeric_type = 'numeric'
        numeric_data_handling = 'normal'
        numeric_scenario = 4

        inference_set, obs_dim = inference_dataset(predict_region, pm_type, horizon, period_version, rm_region,
                                                   exp_name, sampling, lag, data_dir, pca_dim, numeric_type,
                                                   numeric_data_handling, numeric_scenario, seed=999,
                                                   co2_load=False, type=inference_type)

        print(esv_year, f'inference_set: {inference_set.__len__()} took {time.time() - now:.2f} s')

        for esv_y in esv_year:
            e = deepcopy(e)
            e['esv_year'] = esv_y
            net.load_state_dict(model_data['model_weights'][esv_y])

            # inference -> need to set device
            score, orig_pred, pred, label = inference(net, inference_set, pm_type, model_name, model_type, is_reg,
                                                      batch_size=1000, seed=999, device=device)

            for k in ['f1', 'accuracy', 'hit', 'pod', 'far']:
                e[f'val_{k}'] = score[k]
            series_list.append(e)
        df = pd.concat(series_list, axis=1)
        df = df.T.reset_index().drop(['index'], axis=1)
        tmp_df.append(df)

        if debug and idx == 10:
            break

    tmp_df = pd.concat(tmp_df)
    tmp_df.reset_index(drop=True, inplace=True)
    tmp_df.to_excel(os.path.join(root_dir, f'{region}_{inference_type}inference.xlsx'), engine='xlsxwriter')
    return tmp_df


