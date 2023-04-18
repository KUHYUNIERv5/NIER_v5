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
    args = parser.parse_args()

    make_v3_dataset(args.data_dir)