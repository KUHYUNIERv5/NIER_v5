#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2023/03/24 2:33 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : timecheck.py
# @Software  : PyCharm
import numpy as np
import os
from src.utils import load_data, read_yaml, get_region_grid, get_logger
from sklearn.model_selection import ParameterGrid
import datetime
import time


def get_est_time(logger, settings):
    root_dir = '/workspace/results/v5_phase2/R4_64/'

    tmp_dir = os.path.join(root_dir, 'tmp')
    result_dir = os.path.join(root_dir, 'results')
    model_dir = os.path.join(root_dir, 'models')

    id_list = [load_data(os.path.join(tmp_dir, file)) for file in os.listdir(tmp_dir)]
    if len(id_list) != 0:
        id_list = np.concatenate(id_list)

    obj = {
        'debug_mode': [False],
        'predict_region': ['R4_64'],
        'horizon': [3, 4, 5, 6],
        'pm_type': ['PM10', 'PM25']
    }
    obj_list = list(ParameterGrid(obj))
    param_list = []

    for param in obj_list:
        grids = get_region_grid('R4_64', settings, param['pm_type'].lower())
        for grid in grids:
            for key in param.keys():
                grid[key] = param[key]
            grid['esv_years'] = settings['esv_years'][grid['periods']]
            param_list.append(grid)

    if len(id_list) != 0:
        done = len(id_list)
    else:
        done = 1
    left = 8 * 14 * len(param_list) - len(id_list)

    str_d = '2023-03-23 15:44:00'
    since_timestamp = datetime.datetime.strptime(str_d, '%Y-%m-%d %H:%M:%S')

    dt_kst = datetime.datetime.utcnow() + datetime.timedelta(hours=9)

    diff = dt_kst - since_timestamp
    remain_sec = (diff.total_seconds() // done) * left
    remain_time = datetime.timedelta(seconds=remain_sec)

    return datetime.timedelta(seconds=diff.total_seconds()), remain_time, done, left


#     print('duration: ', datetime.timedelta(seconds=diff.total_seconds()))
#     print('est remain_time: ', remain_time)

# Print iterations progress
def printProgressBar(iteration, total, remain, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r",
                     end=' '):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix} {remain}', end=" ", flush=True)
    # Print New Line on Complete
    if iteration == total:
        print()


if __name__ == "__main__":
    settings = read_yaml('./data_folder/settings.yaml')
    logger = get_logger('check_time', '.')

    while True:
        duration, remain, done, left = get_est_time(logger, settings)

        printProgressBar(done, left, remain)

        time.sleep(5)