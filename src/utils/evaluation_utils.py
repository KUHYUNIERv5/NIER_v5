#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/10/26 1:47 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : evaluation_utils.py
# @Software  : PyCharm

from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error, mean_absolute_error

def roc_auc(y_true, y_score):
    return 100 * roc_auc_score(y_true=y_true, y_score=y_score)

def mse(y_true, y_pred):
    return mean_squared_error(y_true=y_true, y_pred=y_pred)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count