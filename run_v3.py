#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2023/04/18 9:58 AM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : run_v3.py
# @Software  : PyCharm

import os
import time
import pandas as pd
import numpy as np

from src.dataset.dataloader_v3 import make_v3_dataset, V3Dataset
from src.utils import load_data, save_data, concatenate, AverageMeter, set_random_seed, concatenate
from sklearn.metrics import confusion_matrix

from tqdm.auto import tqdm
from copy import deepcopy

from torch.utils.data import Dataset, DataLoader
import torch


class V3_Runner:
    def __init__(self, region='R4_68',
                 device='cpu',
                 data_dir='/workspace/R5_phase2/',
                 root_dir='/workspace/results/v5_phase2/',
                 inference_type=2022,
                 pm_type='PM10',
                 horizon=3,
                 validation_days=14,
                 ensemble=True,
                 num_ensemble_models=10,
                 debug=True):
        super().__init__()

        self.region = region
        self.device = device
        self.data_dir = data_dir
        self.root_dir = root_dir
        self.inference_type = inference_type
        self.debug = debug
        self.pm_type = pm_type
        self.horizon = horizon
        self.validation_days = validation_days
        self.ensemble = ensemble
        self.num_ensemble_models = num_ensemble_models

        self.root_dir = os.path.join(root_dir, region)

        self.model_dir = os.path.join(root_dir, 'models')
        self.tmp_dir = os.path.join(root_dir, 'tmp')

        self.csv_dir = os.path.join(root_dir, 'id_list.csv')
        exp_settings = pd.read_csv(self.csv_dir)

        str_expr = f"(predict_region == '{region}') and (pm_type == '{pm_type}') and (horizon == {horizon})"
        self.exp_settings = exp_settings.query(str_expr)

        self.dataset_bundles, self.model_pools, self.max_length = self._load_data_model()

    def _load_data_model(self):
        now = time.time()
        dataset_bundles = []
        model_pools = []
        for i in tqdm(range(len(self.exp_settings))):
            e = self.exp_settings.iloc[i]
            esv_year = 2021

            model_file_name = f'{e.id}.pkl'
            model_data = load_data(os.path.join(self.model_dir, model_file_name))
            net = model_data['network']
            net.load_state_dict(model_data['model_weights'][esv_year])

            model_pools.append(deepcopy(net))

            dataset_args = dict(
                predict_region=self.region,
                pm_type=e.pm_type,
                rm_region=e.rm_region,
                representative_region=e.representative_region,
                period_version=e.period_version,
                data_path=self.data_dir,
                lag=e.lag,
                max_lag=7,
                horizon=e.horizon,
                validation_days=self.validation_days
            )

            dataset = V3Dataset(**dataset_args)
            dataset_bundles.append(dataset)
        max_length = dataset_bundles[0].max_length
        print(f'Model & Dataset load took: {time.time() - now:.2f} s')
        return dataset_bundles, model_pools, max_length

    def _evaluation(self, y_list, pred_list):
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

    def run_v3_batch(self, data, net, scale, mean, is_reg, device):
        (obs_batch, fnl_batch, num_batch, y_batch, y_orig_batch, y_cls_batch, horizon_batch), _, prediction_date = data

        pred_list = None
        y_list = None
        original_pred_list = None

        net.eval()

        obs_batch = obs_batch.to(device).float()
        obs_batch = obs_batch.permute(0, 2, 1).contiguous()
        fnl_batch = fnl_batch.to(device).float()
        fnl_batch = fnl_batch.permute(0, 2, 1).contiguous()
        num_batch = num_batch.to(device).float()

        point_num = None
        if horizon_batch.shape[-1] != 1:
            point_num = horizon_batch.to(device).float()

        if is_reg:
            y = y_batch.to(device).float()
            orig_y = y_orig_batch.to(device).float()
        else:
            y = y_cls_batch.to(device).long()

        x_pred = net(obs_batch, fnl_batch, num_batch, point_num)

        if is_reg:
            x_pred = x_pred.squeeze(-1) if x_pred.shape[0] == 1 else x_pred.squeeze()
            original_pred_list = x_pred.detach().clone().cpu().numpy()
            y_list = concatenate(y_list, orig_y.detach().clone().cpu().numpy())
            pred_list = concatenate(pred_list, x_pred.detach().clone().cpu().numpy() * scale + mean)
        else:
            preds = x_pred.argmax(dim=1)
            y_list = concatenate(y_list, y.detach().clone().cpu().numpy())
            pred_list = concatenate(pred_list, preds.detach().clone().cpu().numpy())
            original_pred_list = concatenate(original_pred_list, x_pred.detach().clone().cpu().numpy(), axis=0)

        return original_pred_list, pred_list, y_list, prediction_date

    def _thresholding(self, array, thresholds):
        y = array.squeeze()
        y_cls = np.zeros_like(y)
        for i, threshold in enumerate(thresholds):
            y_cls[y > threshold] = i + 1
        return y_cls

    def run_v3(self):

        pass


def main():
    best_config = './data_folder/'
    pass

# if __name__ == "__main__":
