#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/10/26 1:46 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : vector_utils.py
# @Software  : PyCharm

import torch
import numpy as np
from itertools import groupby


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

def one_hot(array):
    unique, inverse = np.unique(array, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    return onehot


def easycat(array1, array2, axis=0):
    # assert isinstance(array2, np.ndarray)
    if array1 is not None:
        if isinstance(array1, np.ndarray):
            return np.concatenate((array1, array2), axis=axis)
        elif isinstance(array1, torch.Tensor):
            return torch.concat((array1, array2), dim=axis)
    else:
        return array2

def concatenate_columns(cols):
    conc = None

    for k in cols.keys():
        if len(cols[k].shape) < 2:
            conc = concatenate(conc, np.expand_dims(cols[k], 1), axis=1)
        else:
            conc = concatenate(conc, cols[k], axis=1)

    return conc

def concatenate(array1, array2, axis=0):
    assert isinstance(array2, np.ndarray)
    if array1 is not None:
        assert isinstance(array1, np.ndarray)
        return np.concatenate((array1, array2), axis=axis)
    else:
        return array2