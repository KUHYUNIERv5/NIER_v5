#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2023/03/12 5:55 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : train_validation.py
# @Software  : PyCharm

from src.trainer import BasicTrainer
from src.dataset import NIERDataset
from src.utils import save_data, read_yaml, load_data, get_region_grid, get_best_hyperparam, merge_two_dicts
from sklearn.model_selection import ParameterGrid

from itertools import product
import torch.multiprocessing as mp
import torch
from tqdm.auto import tqdm
import numpy as np
import uuid
import argparse
import os


def arg_config(predict_region, pm_type='PM10'):  # ['PM10', 'PM25'] # [3,4,5,6]
    global data_dir, numeric_scenario, numeric_type, numeric_data_handling, model_ver, seed, lr, dropout, pca_dim
    # top priority hyperparams
    co2_load = False

    # fixed params
    root_dir = '/workspace/local/R5_phase2_saves/'
    data_dir = '../dataset/d5_phase2'  # setting 필요'../../dataset/d5_phase2'
    numeric_scenario = 4
    numeric_type = 'numeric'
    numeric_data_handling = 'normal'
    model_ver = 'v2'
    n_epochs = 1  # 50
    batch_size = 128
    lr = 0.001
    dropout = .1

    if numeric_scenario == 0 and model_ver == 'v1':
        num_setting = 'r4'
    elif numeric_scenario == 3:
        num_setting = 'r5'
    else:
        num_setting = 'r6'

    seed = 999

    # outer hyperparams
    settings = read_yaml('./data_folder/settings.yaml')
    grids = get_region_grid(predict_region, settings, pm_type.lower())
    data_grid = dict(
        sampling=['normal', 'oversampling'],
        lag=[1, 2, 3, 4, 5, 6, 7]
    )
    data_grids = list(ParameterGrid(data_grid))

    li = []
    for grid in grids:
        for data_grid in data_grids:
            li.append(merge_two_dicts(grid, data_grid))
    grids = list(map(dict, set(tuple(sorted(sub.items())) for sub in li)))

    return n_epochs, batch_size, co2_load, grids, settings

def prepare_trainset(co2_load, predict_region, pm_type, horizon, period_version, rm_region, esv_years, exp_name,
                     sampling, lag, data_dir, pca_dim, numeric_type, numeric_data_handling='normal',
                     numeric_scenario=4, seed=999):
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
        test_period_version='tmp',
        esv_year=2021,  # early stopping validation year
        co2_load=co2_load,
        rm_region=rm_region,
        exp_name=exp_name
    )

    trainset = NIERDataset(**dataset_args)
    dataset_args['data_type'] = 'valid'
    validsets = []
    for esv_year in esv_years:
        dataset_args['esv_year'] = esv_year
        validset = NIERDataset(**dataset_args)
        validsets.append(validset)

    if co2_load:
        obs_dim = dataset_args['pca_dim']['obs'] + 3
    else:
        obs_dim = dataset_args['pca_dim']['obs']

    return trainset, validsets, obs_dim


def run_trainer(train_set, valid_sets, predict_region, pm_type, horizon, obs_dim, esv_years, sampling, lag, is_reg,
                model_name, model_type, n_epochs, dropout, device, optimizer_name="SGD",
                objective_name="CrossEntropyLoss", numeric_type='numeric'):


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
        log_flag=True
    )

    trainer_args['device'] = device

    trainer = BasicTrainer(**trainer_args)

    train_val_dict = dict(
        model_args=dict(
            obs_dim=obs_dim,
            fnl_dim=pca_dim['fnl'],
            num_dim=pca_dim[numeric_type],
            lag=lag,
        ),
        optimizer_args={
            'lr': lr,
            'momentum': 0.9,
            'weight_decay': 1e-5,
            'nesterov': True
        },
        param_args={
            'horizon': horizon,
            'sampling': sampling,
            'region': predict_region
        },
        objective_args={},
        batch_size=64
    )

    net, best_model_weights, return_dict = trainer.esv_train(train_set, valid_sets, esv_years, **train_val_dict)

    return net, best_model_weights, return_dict


def main(device, pm_type, horizon, predict_region):
    result_grid = []
    model_grid = []

    n_epochs, batch_size, co2_load, grids, settings = arg_config(predict_region, pm_type)

    for grid in tqdm(grids): # outer grid
        representative_region = grid['representative_region']
        period_version = grid['periods']
        rm_region = grid['remove_regions']
        esv_years = settings['esv_years'][period_version]
        exp_name = os.path.join(predict_region,
                                f'{predict_region}_{representative_region}_period_{period_version}_rmgroup_{rm_region}')

        inner_grid = dict(
            is_reg=[True, False],
            model_name=['RNN', 'CNN'],
            model_type=['single', 'double']
        )
        inner_grids = list(ParameterGrid(inner_grid))

        train_set, valid_sets, obs_dim = prepare_trainset(co2_load, predict_region, pm_type, horizon,
                                                          period_version, rm_region, esv_years, exp_name,
                                                          grid['sampling'], grid['lag'], data_dir,
                                                          pca_dim, numeric_type, seed=seed)

        for inner_grid in inner_grids:
            run_type = 'regression' if inner_grid['is_reg'] else 'classification'
            net, best_model_weights, return_dict = run_trainer(train_set, valid_sets, predict_region, pm_type,
                                                               horizon, obs_dim, esv_years, grid['sampling'],
                                                               grid['lag'], inner_grid['is_reg'],
                                                               inner_grid['model_name'], inner_grid['model_type'],
                                                               n_epochs, dropout, device)
            setting_id = uuid.uuid4()

            settings = {
                'id' : setting_id,
                'hyperparams': dict(
                    predict_region=predict_region,
                    pm_type=pm_type,
                    horizon=horizon,
                    representative_region=representative_region,
                    period_version=period_version,
                    rm_region=rm_region,
                    esv_years=esv_years,
                    exp_name=exp_name,
                    lag=grid['lag'],
                    sampling=grid['sampling'],
                    run_type=run_type,
                    model=inner_grid['model_name'],
                    model_type=inner_grid['model_type'],
                ),

                'results': dict(
                    y_labels=return_dict['y_labels'],
                    val_preds=return_dict['best_preds'],
                    val_orig_preds=return_dict['']
                )
            }
            model_weights = {
                'id': setting_id,
                'network': net,
                'model_weights': best_model_weights
            }


def multi_gpu(params):
    gpu_idx = semaphore.pop()
    device = 'cuda:%d' % gpu_idx if torch.cuda.is_available() else 'cpu'
    print(f'Process using {device}')
    score_dict = dict()
    model_dict = dict()

    main(device, params['pm_type'], params['horizon'], params['predict_region'])

    # run_info = dict(
    #     region=dataset_args['predict_location_id'],
    #     pm=dataset_args['predict_pm'],
    #     horizon=dataset_args['horizon'],
    #     lag=dataset_args['lag'],
    #     sampling=dataset_args['sampling'],
    #     num_scen=dataset_args['numeric_scenario'],
    #     model_ver=trainer_dict['model_ver'],
    #     model=trainer_dict['model_name'],
    #     c_or_r=trainer_dict['is_reg'],
    #     layer=trainer_dict['model_type'],
    #     run_epoch=trainer_dict['n_epochs'],
    #     best_epoch=return_dict['best_epoch'],  # 0부터 count
    #     lr=lr,
    # )
    #
    # score_dict[idx] = dict(
    #     run_info=run_info,
    #     best_valid_scores=return_dict['best_score'],
    #     all_train_scores=return_dict['train_score_list'],
    #     all_valid_scores=return_dict['val_score_list'],
    #     train_loss=return_dict['train_loss_list'],
    #     valid_loss=return_dict['val_loss_list'],
    #     best_val_pred_orig=return_dict['best_orig_pred'],
    #     best_val_pred=return_dict['best_pred'],
    #     y_label=return_dict['y_label']
    # )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='retrain arg')

    parser.add_argument('--debug', '-d', action="store_true")
    parser.add_argument('--region', help='input region', default='R4_62')

    args = parser.parse_args()

    gpu_idx_list = np.arange(8)
    obj = {
        'debug_mode': [args.debug],
        'predict_region': [args.region],
        'horizon': [3,4,5,6],
        'pm_type': ['PM10', 'PM25']
    }
    param_list = list(ParameterGrid(obj))
    manager = mp.Manager()
    semaphore = manager.list(gpu_idx_list)
    results_dict = manager.dict()
    models_dict = manager.dict()
    pool = mp.Pool(processes=len(gpu_idx_list))
    pool.map(multi_gpu, param_list)
    pool.close()
    pool.join()
# period_version, rm_region, esv_years, sampling, lag, is_reg, model_name, model_type, exp_name, n_epochs, dropout, device,
