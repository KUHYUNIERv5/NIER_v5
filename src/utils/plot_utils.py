#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/10/21 2:29 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : plot_utils.py
# @Software  : PyCharm

import numpy as np
import matplotlib.pyplot as plt


def plot_pca_explained_variances(pca, type, idx_list=None, test_year=21):
    if idx_list is None:
        idx_list = [64, 128, 256, 512]
    y_tick = [.2, .4, .6, .8]
    cumsum = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(15, 10))
    plt.plot(cumsum)

    plt.title(f'D4 : 2017-20{test_year} {type} PCA explained variance plot', fontsize=26)
    for idx in idx_list:
        plt.axhline(y=cumsum[idx], ls='--', color='r', linewidth=1)
        plt.axvline(x=idx, color='r', linewidth=1)
        y_tick.append(cumsum[idx])
        plt.plot(idx, cumsum[idx], 'ro')
    plt.xlabel('Number of components', fontsize=14)
    plt.ylabel('Cumulative explained variance', fontsize=14)
    plt.xticks([0., 64., 128., 256., 512., 1024.], fontsize=14)
    plt.yticks(y_tick, fontsize=14)
    plt.show()
