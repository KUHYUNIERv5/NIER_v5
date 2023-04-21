#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2023/04/05 2:02 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : retrieve_result.py
# @Software  : PyCharm
<<<<<<< HEAD
from src.evaluation import get_region_result, get_region_resultv2
=======
from src.evaluation import get_region_result
>>>>>>> 78f190460c13ecdfc174561926081d934a74cc0c
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='retrieve experiment results')

    parser.add_argument('--exp_dir', '-d', type=str, help='directory to save results',
<<<<<<< HEAD
                        default='/workspace/code/R5_phase2_saves')
    parser.add_argument('--region', '-r', type=str)

    args = parser.parse_args()
    _ = get_region_resultv2(exp_dir=args.exp_dir, region=args.region)
=======
                        default='/home/pink/dust/external_drive/dust_prediction_phase_2_multiGPU/')
    parser.add_argument('--region', '-r', type=str)

    args = parser.parse_args()
    _ = get_region_result(exp_dir=args.exp_dir, region=args.region)
>>>>>>> 78f190460c13ecdfc174561926081d934a74cc0c
