#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2023/03/12 5:55 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : train_validation.py
# @Software  : PyCharm

from src.trainer import BasicTrainer
from src.dataset import NIERDataset
from src.utils import save_data, read_yaml, get_region_grid, rmdir, load_data
from sklearn.model_selection import ParameterGrid
from pathlib import Path

from itertools import chain
import torch.multiprocessing as mp
import torch
from tqdm.auto import tqdm
import uuid
import argparse
import os
import pandas as pd
import numpy as np
import datetime


def arg_config(debug_mode=False):  # ['PM10', 'PM25'] # [3,4,5,6]
    global numeric_scenario, numeric_type, numeric_data_handling, model_ver, seed, lr, dropout, pca_dim
    # fixed params
    numeric_scenario = 4
    numeric_type = 'numeric'
    numeric_data_handling = 'normal'
    model_ver = 'v2'
    n_epochs = 2 if debug_mode else 50
    batch_size = 256
    lr = 0.001
    dropout = .1
    pca_dim = dict(  # 건드리면 안됨
        obs=256,
        fnl=512,
        wrf=128,
        cmaq=512,
        numeric=512
    )
    seed = 999

    if debug_mode:
        data_grid = dict(
            sampling=['normal'],
            lag=[1]
        )
    else:
        data_grid = dict(
            sampling=['normal', 'oversampling'],
            lag=[1, 2, 3, 4, 5, 6, 7]
        )
    grids = list(ParameterGrid(data_grid))
    return n_epochs, batch_size, grids


def prepare_trainset(predict_region, pm_type, horizon, period_version, rm_region, esv_years, exp_name,
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
        test_period_version='v2',
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
    dataset_args['data_type'] = 'test'
    testset = NIERDataset(**dataset_args)

    if co2_load:
        obs_dim = dataset_args['pca_dim']['obs'] + 3
    else:
        obs_dim = dataset_args['pca_dim']['obs']

    return trainset, validsets, testset, obs_dim


def run_trainer(train_set, valid_sets, test_set, predict_region, pm_type, horizon, obs_dim, esv_years, sampling, lag,
                is_reg, model_name, model_type, n_epochs, dropout, device, optimizer_name="SGD",
                objective_name="CrossEntropyLoss", numeric_type='numeric', batch_size=64):
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
        log_flag=False
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
        batch_size=batch_size
    )

    if run_cv:
        # cv result
        net, best_model_weights, kfold_models, cv_f1_score, cv_results = trainer.cross_validate(train_set,
                                                                                                **train_val_dict)
        val_dict = {}
        test_dict = {}
    else:
        # esv result
        net, best_model_weights, val_dict = trainer.esv_train(train_set, valid_sets, esv_years, **train_val_dict)

        test_dict = dict()
        for esv_year in esv_years:
            net.load_state_dict(best_model_weights[esv_year])
            test_score, test_orig_pred, test_pred_score, test_label_score = trainer.test(test_set, net, 128,
                                                                                         train_val_dict[
                                                                                             'optimizer_args'],
                                                                                         train_val_dict[
                                                                                             'objective_args'])
            test_dict[esv_year] = {
                'test_result': test_score,
                'test_orig_pred': test_orig_pred,
                'test_pred_score': test_pred_score,
                'test_label_score': test_label_score
            }
        cv_f1_score = 0
        cv_results = 0
    return net, best_model_weights, val_dict, test_dict, cv_f1_score, cv_results


def main(device, pm_type, horizon, predict_region, representative_region, period_version, rm_region, esv_years,
         semaphore, debug_mode=False, resume=False):
    id_list = []
    n_epochs, batch_size, grids = arg_config(debug_mode)
    result_dir = os.path.join(root_dir, 'results')
    model_dir = os.path.join(root_dir, 'models')
    tmp_dir = os.path.join(root_dir, 'tmp')

    for grid in tqdm(grids):  # outer grid
        exp_name = os.path.join(predict_region,
                                f'{predict_region}_{representative_region}_period_{period_version}_rmgroup_{rm_region}')

        if debug_mode:
            inner_grid = dict(
                is_reg=[True],
                model_name=['RNN'],
                model_type=['single']
            )
        else:

            inner_grid = dict(
                is_reg=[True, False],
                model_name=['RNN', 'CNN'],
                model_type=['single', 'double']
            )
        inner_grids = list(ParameterGrid(inner_grid))

        train_set, valid_sets, test_set, obs_dim = prepare_trainset(predict_region, pm_type, horizon,
                                                                    period_version, rm_region, esv_years, exp_name,
                                                                    grid['sampling'], grid['lag'], data_dir,
                                                                    pca_dim, numeric_type, seed=seed)

        for inner_grid in inner_grids:

            # configuration ID
            setting_id = uuid.uuid4()

            run_type = 'regression' if inner_grid['is_reg'] else 'classification'

            is_included = check_condition(resume_df, grid['sampling'], grid['lag'], inner_grid['model_name'],
                                          inner_grid['model_type'], run_type)

            if resume and is_included:
                continue

            net, best_model_weights, val_dict, test_dict, cv_f1_score, cv_results \
                = run_trainer(train_set, valid_sets, test_set, predict_region, pm_type, horizon, obs_dim, esv_years,
                              grid['sampling'], grid['lag'], inner_grid['is_reg'], inner_grid['model_name'],
                              inner_grid['model_type'], n_epochs, dropout, device, batch_size=batch_size)

            if run_cv:
                results = {
                    'id': setting_id,
                    'cv_score': cv_f1_score,
                    'cv_results': cv_results
                }
                model_weights = {
                    'id': setting_id,
                    'network': net.cpu(),
                    'model_weights': best_model_weights
                }

            else:
                results = {
                    'id': setting_id,
                    'val_results': val_dict,
                    'test_results': test_dict
                }

                model_weights = {
                    'id': setting_id,
                    'network': net.cpu(),
                    'model_weights': best_model_weights
                }

            save_data(results, result_dir, f'{setting_id}.pkl')
            save_data(model_weights, model_dir, f'{setting_id}.pkl')

            ids = dict(
                id=setting_id,
                predict_region=predict_region,
                pm_type=pm_type,
                horizon=horizon,
                representative_region=representative_region,
                period_version=period_version,
                rm_region=rm_region,
                esv_years=esv_years,
                # exp_name=exp_name, # 필요없는 정보
                lag=grid['lag'],
                sampling=grid['sampling'],
                run_type=run_type,
                model=inner_grid['model_name'],
                model_type=inner_grid['model_type'],
            )
            id_list.append(ids)

            save_data(id_list, tmp_dir, f'id_list_{device}_{semaphore}.pkl')
    return id_list


def check_condition(df, oversampling, lag, model_name, model_type, run_type, representative_region, period_version,
                    rm_region):
    condition = (df['sampling'] == oversampling) & (df['lag'] == lag) & (df['run_type'] == run_type) & (
            df['model'] == model_name) & (df['model_type'] == model_type) & (
                        df['representative_region'] == representative_region) & (
                        df['period_version'] == period_version) & (df['rm_region'] == rm_region)

    # Check if condition is included in DataFrame
    return condition.any()


def multi_gpu(params):
    gpu_idx = gpus.pop()
    sema = semaphore.pop()
    device = 'cuda:%d' % gpu_idx if torch.cuda.is_available() else 'cpu'
    print(f'Process using {device}')
    # result_grid, model_grid,
    id_list = main(device, params['pm_type'], params['horizon'], params['predict_region'], \
                   representative_region=params['representative_region'], \
                   period_version=params['periods'], rm_region=params['remove_regions'], \
                   esv_years=params['esv_years'], debug_mode=params['debug_mode'], semaphore=sema,
                   resume=params['resume'])

    print(f'Process end {device}')

    gpus.append(gpu_idx)
    semaphore.append(sema)
    return_id_lists.append(id_list)


def reset_all():
    if not os.path.exists(os.path.join(root_dir, 'id_list.csv')):
        result_dir = os.path.join(root_dir, 'results')
        model_dir = os.path.join(root_dir, 'models')
        tmp_dir = os.path.join(root_dir, 'tmp')

        if os.path.exists(result_dir):
            rmdir(Path(result_dir))
        if os.path.exists(model_dir):
            rmdir(Path(model_dir))
        if os.path.exists(tmp_dir):
            rmdir(Path(tmp_dir))

        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(tmp_dir, exist_ok=True)


def resume_train():
    tmp_dir = os.path.join(root_dir, 'tmp')
    file_list = []
    for file in os.listdir(tmp_dir):
        if file.endswith('.pkl'):
            file_list.append(load_data(os.path.join(tmp_dir, file)))
            print(os.path.join(tmp_dir, file))
    file_list = np.concatenate(file_list)
    for i in range(len(file_list)):
        file_list[i]['id'] = str(file_list[i]['id'])
    file_list = file_list.tolist()
    file_df = pd.DataFrame(file_list)
    save_data(file_list, tmp_dir, f'resumed_list.pkl')

    return file_df, file_list


if __name__ == "__main__":
    global return_id_lists, root_dir, data_dir, co2_load, run_cv, resume_list, resume_df
    import logging

    now = datetime.datetime.now()
    try:
        os.makedirs('../logs', exist_ok=True)
        print("Directory '%s' created successfully" % '../logs')
    except OSError as error:
        print("Directory '%s' can not be created")

    logging.basicConfig(filename=f'../logs/error_{now.strftime("%Y-%m-%d %H:%M:%S")}.log', level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='retrain arg')

    parser.add_argument('--debug', '-d', action="store_true")
    parser.add_argument('--reset', '-r', action="store_true")
    parser.add_argument('--run_cv', '-rc', action="store_true")
    parser.add_argument('--resume', '-rs', action="store_true")
    parser.add_argument('--gpu_list', '-gl', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument('--region', help='input region', default='R4_62')
    parser.add_argument('--co2', '-c', type=bool, help='load co2 data or not', default=False)
    parser.add_argument('--root_dir', '-rd', type=str, help='directory to save results',
                        default='/workspace/code/R5_phase2_saves')
    parser.add_argument('--data_dir', '-dd', type=str, default='/workspace/data/NIERDataset/R5_phase2/data_folder')

    args = parser.parse_args()

    current_time = datetime.datetime.utcnow() + datetime.timedelta(hours=9)

    debug = args.debug
    region = args.region
    co2_load = args.co2
    root_dir = args.root_dir
    run_cv = args.run_cv
    print(
        f"training starting at {current_time.strftime('%Y-%m-%d %H:%M:%S')} with run cv = {run_cv} debug mode {debug}")
    if debug:
        root_dir = os.path.join(root_dir, 'debugging')
    else:
        root_dir = os.path.join(root_dir, region)
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    if run_cv:
        root_dir = os.path.join(root_dir, f'cv')

    os.makedirs(root_dir, exist_ok=True)
    resume_list = None
    resume_df = None

    data_dir = args.data_dir
    if args.reset and not args.resume:
        reset_all()
    if args.resume:
        resume_df, resume_list = resume_train()
    obj = {
        'resume': [args.resume],
        'debug_mode': [debug],
        'predict_region': [region],
        'horizon': [3, 4, 5, 6],
        'pm_type': ['PM10', 'PM25']
    }
    obj_list = list(ParameterGrid(obj))

    settings = read_yaml('./data_folder/settings.yaml')
    param_list = []

    for param in obj_list:
        grids = get_region_grid(region, settings, param['pm_type'].lower())
        for grid in grids:
            for key in param.keys():
                grid[key] = param[key]
            grid['esv_years'] = settings['esv_years'][grid['periods']]
            param_list.append(grid)
    np.random.shuffle(param_list)

    gpu_idx_list = args.gpu_list

    try:
        manager = mp.Manager()
        gpus = manager.list(gpu_idx_list)
        semaphore = manager.list(np.arange(len(gpu_idx_list)))
        return_id_lists = manager.list()
        pool = mp.Pool(processes=len(gpu_idx_list))
        pool.map(multi_gpu, param_list)
        pool.close()
        pool.join()
    except Exception as error:
        logger.error(error)

    return_id_lists = list(chain(*return_id_lists))
    return_id_lists = pd.DataFrame(return_id_lists)
    if args.resume:
        return_id_lists = pd.concat([resume_df, return_id_lists])
    return_id_lists.to_csv(os.path.join(root_dir, f'id_list.csv'), index=False)
