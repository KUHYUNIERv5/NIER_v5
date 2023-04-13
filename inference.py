#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2023/04/11 10:48 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : inference.py
# @Software  : PyCharm

from src.evaluation.inference_module import inference_on_validset

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='retrieve experiment results')

    parser.add_argument('--debug', '-d', action="store_true")
    parser.add_argument('--data_dir', '-dd', type=str, help='directory to save results',
                        default='/workspace/R5_phase2/')
    parser.add_argument('--root_dir', '-rd', type=str, default='/workspace/results/v5_phase2/')
    parser.add_argument('--region', '-r', type=str)
    parser.add_argument('--device', '-d', type=str, default='cpu')
    parser.add_argument('--inference_type', '-i', type=int, default=2021)

    args = parser.parse_args()

    inference_type = args.inference_type
    _ = inference_on_validset(region=args.region, device=args.device, data_dir=args.data_dir,
                          root_dir=args.root_dir, inference_type=inference_type, debug=args.debug)
