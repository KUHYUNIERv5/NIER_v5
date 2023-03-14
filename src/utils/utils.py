#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/10/21 2:28 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : utils.py
# @Software  : PyCharm


import json
import os
import pickle

import numpy as np
import torch
import random
import logging
import yaml

from .vector_utils import all_equal, concatenate
from sklearn.model_selection import ParameterGrid

def merge_two_dicts(x, y):
    z = x.copy()   # start with keys and values of x
    z.update(y)    # modifies z with keys and values of y
    return z

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

def get_region_grid(region, settings, pm_type):
    hyperparam = get_best_hyperparam(region, settings)

    final_grid = None
    for k in hyperparam.keys():
        hyperparam[k]
        final_grid = np.array(list(ParameterGrid(hyperparam[k][pm_type])))

    res = list(map(dict, set(tuple(sorted(sub.items())) for sub in final_grid)))
    return res


# Function for repeatability
def set_random_seed(seed_value):
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.random.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def not_null(df, col):
    unique_list = df[~df[col].isnull()][col].unique()
    return unique_list


def not_null_df(df, col):
    return df[~df[col].isnull()]


def read_yaml(dir):
    with open(dir, 'rb') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        return data


def read_json(path):
    with open(path) as json_file:
        json_data = json.load(json_file)

    return json_data


def write_yaml(dir, data):
    with open(dir, 'w') as f:
        yaml.dump(data, f)


# def read_json(file):
#     tweets = []
#     i = 0
#     for line in open(file, 'r'):
#         tweets.append(json.loads(line))
#         i += 1
#     df = pd.DataFrame(tweets)
#
#     return df

def show_obj_struct(obj, name, tab=''):
    print(f"{tab}- {name}")
    root_keys = list(obj.keys())
    if len(root_keys) > 0:
        for key in root_keys:
            if type(obj[key]) is dict:
                show_obj_struct(obj[key], key, tab + '  ')
            else:
                print(f"{tab + '  '}- {key}")
    else:
        return


def save_data(state, directory, filename='inference_result.pkl'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    with open(filename, 'wb') as f:
        pickle.dump(state, f)


def load_data(file_path):
    with open(file_path, 'rb') as f:
        file = pickle.load(f)
    return file


def join_features(dataset):  # 모든 feature를 하나로 unify
    all_features = []
    for d in dataset:
        for feat in list(d.keys()):
            if feat not in all_features:
                all_features.append(feat)

    return np.array(all_features)


def select_column_list(df, column_list):
    return df[[c for c in df.columns if c in column_list]]


def get_logger(name, file_path):
    log_file_path = f'{file_path}/{name}.log'

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_hander = logging.StreamHandler()
    stream_hander.setFormatter(formatter)
    logger.addHandler(stream_hander)

    file_handler = logging.FileHandler(log_file_path)
    logger.addHandler(file_handler)

    return logger
