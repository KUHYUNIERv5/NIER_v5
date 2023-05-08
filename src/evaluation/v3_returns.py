#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2023/05/04 4:27 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : v3_returns.py
# @Software  : PyCharm
from collections import OrderedDict
import numpy as np
from sklearn.metrics import confusion_matrix

class V3Output(OrderedDict):
    def __init__(self, ensemble_prediction_ls=None, single_prediction_ls=None, label_ls=None, argsort_list=None, argsort_exp_list=None,
                 f1sort_list=None, argsort_topk_list=None, validation_times=None, inference_times=None, total_times=None, make_result=True):
        super().__init__()
        self.ensemble_prediction_ls = ensemble_prediction_ls
        self.single_prediction_ls = single_prediction_ls
        self.label_ls = label_ls
        self.argsort_list = argsort_list
        self.argsort_exp_list = argsort_exp_list
        self.f1sort_list = f1sort_list
        self.argsort_topk_list = argsort_topk_list
        self.validation_times = validation_times
        self.inference_times = inference_times
        self.total_times = total_times

        if make_result:
            self.ensemble_res = self._evaluation(np.array(label_ls, dtype=np.float32),
                                              np.array(ensemble_prediction_ls, dtype=np.float32))
            self.single_res = self._evaluation(np.array(label_ls, dtype=np.float32),
                                            np.array(single_prediction_ls, dtype=np.float32))


    def summary(self):
        keys, counts = np.unique(self.ensemble_prediction_ls, return_counts=True)
        ens_labels = [f'label {k}: {c}' for k, c in zip(keys, counts)]
        keys, counts = np.unique(self.single_prediction_ls, return_counts=True)
        single_label = [f'label {k}: {c}' for k, c in zip(keys, counts)]
        keys, counts = np.unique(self.label_ls, return_counts=True)
        y_label = [f'label {k}: {c}' for k, c in zip(keys, counts)]

        str_output = f"ensemble prediction result info | unique keys: {ens_labels}\n" \
                     f"best model prediction result info | unique keys: {single_label}\n" \
                     f"label info | unique keys: {y_label}\n" \
                     f""



    def _evaluation(self, y_list, pred_list, is_r4=False):
        """
        **필요시 변경해야 함(현재는 단기 팀 세팅 따름)**
        :param y_list: true values
        :param pred_list: predicted values
        :return: object (accuracy, hit, pod, far, f1)
        """

        cfs_matrix = confusion_matrix(y_list, pred_list, labels=[0., 1., 2., 3.])

        accuracy = np.trace(cfs_matrix) / np.sum(cfs_matrix)

        pod = 0. if np.sum(cfs_matrix[2:, 2:]) == 0 else np.sum(cfs_matrix[2:, 2:]) / (np.sum(cfs_matrix[2:, :2]) +
                                                                                       np.sum(cfs_matrix[2:, 2:]))
        far = 1. if np.sum(cfs_matrix[:2, 2:]) + np.sum(cfs_matrix[2:, 2:]) == 0 else np.sum(cfs_matrix[:2, 2:]) / \
                                                                                      (np.sum(
                                                                                          cfs_matrix[:2, 2:]) + np.sum(
                                                                                          cfs_matrix[2:, 2:]))
        f1 = 0. if (pod + (1 - far)) == 0 else (2 * pod * (1 - far)) / (pod + (1 - far))
        hit = 0. if np.sum(cfs_matrix[2]) + np.sum(cfs_matrix[3]) == 0 else (cfs_matrix[2, 2] + cfs_matrix[3, 3]) / \
                                                                            (np.sum(cfs_matrix[2]) + np.sum(
                                                                                cfs_matrix[3]))

        return dict(
            accuracy=accuracy,
            hit=hit,
            pod=pod,
            far=far,
            f1=f1
        )