#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2023/04/05 2:02 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : retrieve_result.py
# @Software  : PyCharm
from src.evaluation import get_region_result
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='retrieve experiment results')

    parser.add_argument('--exp_dir', '-d', type=str, help='directory to save results',
                        default='/home/pink/dust/external_drive/dust_prediction_phase_2_multiGPU/')
    parser.add_argument('--region', '-r', type=str)

    args = parser.parse_args()
    _ = get_region_result(exp_dir=args.exp_dir, region=args.region)