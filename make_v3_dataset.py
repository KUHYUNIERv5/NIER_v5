#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2023/04/18 10:05 AM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : make_v3_dataset.py
# @Software  : PyCharm

from src.dataset.dataloader_v3 import make_v3_dataset
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='make v3 dataset')
    parser.add_argument('--data_dir', '-dd', type=str, help='directory to save results',
                        default='/workspace/R5_phase2/')
    parser.add_argument('--regions', '-r', nargs='+', type=int,
                        default=[59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77])
    args = parser.parse_args()

    make_v3_dataset(args.data_dir, args.regions)
