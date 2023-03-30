#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2023/03/24 2:33 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : timecheck.py
# @Software  : PyCharm
import numpy as np
import os
from src.utils import load_data, read_yaml, get_region_grid, get_logger, merge_dicts, get_all_settings
from sklearn.model_selection import ParameterGrid
import datetime
import time
import argparse

def get_est_time(region, str_d='2023-03-23 15:44:00'):
    root_dir = f'/workspace/results/v5_phase2/{region}/'

    tmp_dir = os.path.join(root_dir, 'tmp')
    result_dir = os.path.join(root_dir, 'results')
    model_dir = os.path.join(root_dir, 'models')

    id_list = [load_data(os.path.join(tmp_dir, file)) for file in os.listdir(tmp_dir)]
    if len(id_list) != 0:
        id_list = np.concatenate(id_list)

    param_list = get_all_settings(region)

    if len(id_list) != 0:
        done = len(id_list)
    else:
        done = 1
    left = len(param_list) - len(id_list)
    print(len(param_list), len(id_list))

    since_timestamp = datetime.datetime.strptime(str_d, '%Y-%m-%d %H:%M:%S')

    dt_kst = datetime.datetime.utcnow() + datetime.timedelta(hours=9)

    diff = dt_kst - since_timestamp
    remain_sec = (diff.total_seconds() // done) * left
    remain_time = datetime.timedelta(seconds=remain_sec)

    return dt_kst.strftime('%Y-%m-%d %H:%M:%S'), datetime.timedelta(
        seconds=diff.total_seconds()), remain_time, done, left


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
    parser = argparse.ArgumentParser(description='retrain arg')
    parser.add_argument('--region', '-r', type=str, help='checking region',
                        default='R4_64')
    parser.add_argument('--start_time', '-t', type=str, default='2023-03-23 15:44:00')

    args = parser.parse_args()

    while True:
        now, duration, remain, done, left = get_est_time(args.region, args.start_time)
        printProgressBar(done, left, remain, prefix=now)
        time.sleep(5)