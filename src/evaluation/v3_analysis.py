#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2023/05/08 11:43 AM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : v3_analysis.py
# @Software  : PyCharm
import numpy as np
import pandas as pd
from src.utils import save_data, load_data
from src.evaluation.v3_module import V3_Runner
from src.evaluation.v3_returns import V3Output
import os


def compute_ranking(output_obj):
    # Find the number of models
    num_models = len(output_obj.argsort_list[0])

    # Initialize an array to hold the ranks of each model across all prediction dates
    model_ranks = np.zeros(num_models)

    # Loop over each prediction date and update the model ranks
    for argsort_indices in output_obj.argsort_list:
        model_ranks[argsort_indices] += np.arange(num_models)

    # Compute the mean rank of each model
    mean_ranks = model_ranks / len(output_obj.argsort_list)

    return mean_ranks


def load_result(region='R4_68',
                pm_type='PM10',
                horizon=4,
                model_type_mod=0,  # normal , if 1 : drop cls_cnn
                mod=0,  # normal, if 1: drop f1 == 1
                num_top_k=5, ):
    return_obj_keys = [
        'ensemble_prediction_ls',  # ensemble prediction results
        'single_prediction_ls',  # best model prediction results
        'label_ls',
        'argsort_list',
        'argsort_exp_list',
        'f1sort_list',
        'argsort_topk_list',
        'validation_times',
        'inference_times',
        'total_times',
    ]

    model_type_keys = ['cls_rnn', 'reg_rnn', 'reg_cnn'] if model_type_mod == 1 else None
    val_f1_limit = 1.1 if mod == 0 else 1.0

    root_dir = f'/workspace/results/v3_r5_2/{region}/'
    save_name = f'pm{pm_type}_horizon{horizon}_modeltype{model_type_mod}_f1limit{mod}_numtopk{num_top_k}'

    res = load_data(os.path.join(root_dir, f'{save_name}.pkl'))
    dic = {return_obj_key: res[return_obj_key] for return_obj_key in return_obj_keys}
    output_obj = V3Output(**dic)

    num_models = len(output_obj.argsort_list[0])
    model_ranks = compute_ranking(output_obj)

    v3_obj = V3_Runner(region=region, pm_type=pm_type, horizon=horizon, model_num=100, )
    v3_obj._make_top_exps(model_type_keys=model_type_keys, val_f1_limit=val_f1_limit)
    exp_settings = v3_obj.exp_settings
    exp_settings['mean rank'] = model_ranks[:num_models - 5]
