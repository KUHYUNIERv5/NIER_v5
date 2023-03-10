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
