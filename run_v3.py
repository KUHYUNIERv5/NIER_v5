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
                 validation_days=7,
                 ensemble=True,
                 num_ensemble_models=10,
                 debug=True,
                 model_num=100,
                 add_r4_models=True):
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
        self.add_r4_models = add_r4_models

        self.root_dir = os.path.join(root_dir, region)
        self.model_dir = os.path.join(root_dir, 'models')
        self.tmp_dir = os.path.join(root_dir, 'tmp')
        self.csv_dir = os.path.join(root_dir, 'id_list.csv')

        str_expr = f"(predict_region == '{region}') and (pm_type == '{pm_type}') and (horizon == {horizon})"
        exp_settings = pd.read_csv(self.csv_dir)
        self.exp_settings = exp_settings.query(str_expr)

        self.exp_idxs = np.random.permutation(len(self.exp_settings))[:model_num] # 임시 코드

        self.dataset_bundles, self.model_pools, self.max_length, self.model_load_times, self.data_load_times = self._load_data_model()


    def _load_data_model(self):
        now = time.time()
        dataset_bundles = []
        model_pools = []
        model_load_times = []
        data_load_times = []

        for i in self.exp_idxs: # 임시 코드임
            now = time.time()
            e = self.exp_settings.iloc[i] # 임시 코드임
            esv_year = 2021 # 임시 코드임

            model_file_name = f'{e.id}.pkl'
            model_data = load_data(os.path.join(self.model_dir, model_file_name))
            net = model_data['network']
            net.load_state_dict(model_data['model_weights'][esv_year])

            model_pools.append(deepcopy(net))
            model_load_times.append(time.time() - now)
            now = time.time()

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
            data_load_times.append(time.time() - now)
        max_length = dataset_bundles[0].max_length
        print(f'Model & Dataset load took: {time.time() - now:.2f} s')
        return dataset_bundles, model_pools, max_length, model_load_times, data_load_times

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

    def run_v3_batch(self, data, net, scale, mean, is_reg):
        (obs_batch, fnl_batch, num_batch, y_batch, y_orig_batch, y_cls_batch, horizon_batch), _, prediction_date = data

        pred_list = None
        y_list = None
        original_pred_list = None

        net.eval()

        obs_batch = obs_batch.to(self.device).float()
        obs_batch = obs_batch.permute(0, 2, 1).contiguous()
        fnl_batch = fnl_batch.to(self.device).float()
        fnl_batch = fnl_batch.permute(0, 2, 1).contiguous()
        num_batch = num_batch.to(self.device).float()

        point_num = None
        if horizon_batch.shape[-1] != 1:
            point_num = horizon_batch.to(self.device).float()

        if is_reg:
            y = y_batch.to(self.device).float()
            orig_y = y_orig_batch.to(self.device).float()
        else:
            y = y_cls_batch.to(self.device).long()

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

    def run_v3_single(self, data, net, scale, mean, is_reg):
        _, (pred_obs, pred_fnl, pred_num, pred_y, pred_y_original, pred_y_cls, horizon_day), prediction_date = data

        net.eval()

        pred_obs = pred_obs.to(self.device).float().unsqueeze(0)
        pred_obs = pred_obs.permute(0, 2, 1).contiguous()
        pred_fnl = pred_fnl.to(self.device).float().unsqueeze(0)
        pred_fnl = pred_fnl.permute(0, 2, 1).contiguous()
        pred_num = pred_num.to(self.device).float().unsqueeze(0)

        point_num = None
        if horizon_day.shape[-1] != 1:
            point_num = horizon_day.to(self.device).float().unsqueeze(0)

        if is_reg:
            y = pred_y.to(self.device).float()
            orig_y = pred_y_original.to(self.device).float()
        else:
            y = pred_y.to(self.device).long()

        pred_obs = torch.cat([pred_obs, pred_obs])
        pred_fnl = torch.cat([pred_fnl, pred_fnl])
        pred_num = torch.cat([pred_num, pred_num])
        point_num = torch.cat([point_num, point_num])
        #     print('v3 single: ', pred_obs.shape, pred_fnl.shape, pred_num.shape, point_num.shape, y.shape)
        x_pred = net(pred_obs, pred_fnl, pred_num, point_num)
        x_pred = x_pred[0].unsqueeze(0)
        #     print('v3 single res', x_pred.shape, x_pred[0].unsqueeze(0).shape)
        #     x_pred = new_forward(net, pred_obs, pred_fnl, pred_num, point_num, is_reg, is_double)
        if is_reg:
            x_pred = x_pred.squeeze(-1) if x_pred.shape[0] == 1 else x_pred.squeeze()
            original_pred = x_pred.detach().clone().cpu().numpy()
            pred = x_pred.detach().clone().cpu().numpy() * scale + mean
            y = orig_y.detach().clone().cpu().numpy()
        else:
            preds = x_pred.argmax(dim=1)
            pred = preds.detach().clone().cpu().numpy()
            y = y.detach().clone().cpu().numpy()
            original_pred = x_pred.detach().clone().cpu().numpy()
        #     print('v3 single ret', original_pred.shape, pred.shape, y.shape)
        return original_pred, pred, y, prediction_date

    def _thresholding(self, array, thresholds):
        y = array.squeeze()
        y_cls = np.zeros_like(y)
        for i, threshold in enumerate(thresholds):
            y_cls[y > threshold] = i + 1
        return y_cls

    def run_v3(self):
        final_pred = []
        final_label = []

        validation_times = []
        inference_times = []
        total_times = []

        for i in tqdm(range(self.max_length)):
            f1_list = []
            now = time.time()

            # validation
            for j, (model, data) in enumerate(zip(self.model_pools, self.dataset_bundles)):
                e = self.exp_settings.iloc[self.exp_idxs[j]]

                representative_region = e.representative_region
                period_version = e.period_version
                rm_region = e.rm_region
                pm_type = e.pm_type
                sampling = e.sampling
                horizon = e.horizon
                lag = e.lag
                model_name = e.model
                model_type = e.model_type
                is_reg = True if e.run_type == 'regression' else False

                file_name = f'{self.region}_{representative_region}_period_{period_version}_rmgroup_{rm_region}_v3'

                test_orig_pred, test_pred, test_label, prediction_date = self.run_v3_batch(data[i], model, data.scale,
                                                                                      data.mean, is_reg)

                if is_reg:
                    test_pred_score = self._thresholding(test_pred, data.threshold_dict[pm_type])
                    test_label_score = self._thresholding(test_label, data.threshold_dict[pm_type])
                else:
                    test_pred_score = test_pred
                    test_label_score = test_label

                test_score = self._evaluation(test_label_score, test_pred_score)

                f1_list.append(test_score['f1'])
            validation_times.append(time.time() - now)
            test_now = time.time()

            # test
            if np.mean(f1_list) <= 0:
                argsorts = np.random.permutation(len(self.model_pools))
            else:
                argsorts = np.argsort(f1_list)

            if self.ensemble:
                ensemble_label = None
                ensemble_pred = None
                for j in range(self.num_ensemble_models):
                    data = self.dataset_bundles[argsorts[j]]
                    model = self.model_pools[argsorts[j]]
                    e = self.exp_settings.iloc[self.exp_idxs[argsorts[j]]]
                    is_reg = True if e.run_type == 'regression' else False

                    test_orig_pred, test_pred, test_label, prediction_date = self.run_v3_single(data[i], model, data.scale,
                                                                                           data.mean, is_reg)

                    if is_reg:
                        test_pred_score = self._thresholding(test_pred, data.threshold_dict[pm_type])
                        test_label_score = self._thresholding(test_label, data.threshold_dict[pm_type])
                    else:
                        test_pred_score = test_pred
                        test_label_score = test_label

                    if j == 0:
                        ensemble_label = test_label_score
                    ensemble_pred = concatenate(ensemble_pred, test_pred_score)
                    ensemble_label = concatenate(ensemble_label, test_label_score)

                final_pred.append(ensemble_pred)
                final_label.append(ensemble_label)
            else:
                data = self.dataset_bundles[argsorts[0]]
                model = self.model_pools[argsorts[0]]
                e = self.exp_settings.iloc[self.exp_idxs[argsorts[0]]]

                is_reg = True if e.run_type == 'regression' else False
                is_double = True if e.model_type == 'double' else False
                test_orig_pred, test_pred, test_label, prediction_date = self.run_v3_single(data[i], model, data.scale,
                                                                                       data.mean, is_reg)

                if is_reg:
                    test_pred_score = self._thresholding(test_pred, data.threshold_dict[pm_type])
                    test_label_score = self._thresholding(test_label, data.threshold_dict[pm_type])
                else:
                    test_pred_score = test_pred
                    test_label_score = test_label

                final_pred.append(test_pred_score)
                final_label.append(test_label_score)
            inference_times.append(time.time() - test_now)
            total_times.append(time.time() - now)

        return inference_times, total_times


def main():
    best_config = './data_folder/best_model.csv'
    pass

# if __name__ == "__main__":
