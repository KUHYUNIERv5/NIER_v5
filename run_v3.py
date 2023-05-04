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
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch


class V3_Runner:
    def __init__(self, region='R4_68',
                 device='cpu',
                 data_dir='/workspace/R5_phase2/',
                 root_dir='/workspace/results/v5_phase2/',
                 r4_dir='/workspace/results/v3_um_r4to_r5',
                 cmaq_dir='/workspace/results/v3_cmaq/',
                 inference_type=2022,
                 pm_type='PM10',
                 horizon=3,
                 validation_days=7,
                 ensemble=True,
                 num_ensemble_models=10,
                 debug=True,
                 num_model_per_type=25,
                 add_r4_models=True,
                 add_cmaq_model=True):
        super().__init__()

        self.region = region
        self.device = device
        self.data_dir = data_dir
        self.root_dir = root_dir
        self.r4_dir = r4_dir
        self.cmaq_dir = cmaq_dir
        self.inference_type = inference_type
        self.debug = debug
        self.pm_type = pm_type
        self.horizon = horizon
        self.validation_days = validation_days
        self.ensemble = ensemble
        self.num_ensemble_models = num_ensemble_models
        self.num_model_per_type = num_model_per_type
        self.add_r4_models = add_r4_models
        self.add_cmaq_model = add_cmaq_model

        self.root_dir = os.path.join(root_dir, region)
        self.model_dir = os.path.join(self.root_dir, 'models')
        self.tmp_dir = os.path.join(self.root_dir, 'tmp')
        self.csv_dir = os.path.join(self.root_dir, 'id_list.csv')

        # str_expr = f"(predict_region == '{region}') and (pm_type == '{pm_type}') and (horizon == {horizon})"
        # exp_settings = pd.read_csv(self.csv_dir)
        # self.exp_settings = exp_settings.query(str_expr)
        self.exp_settings = self._make_top_exps()
        self.model_num = len(self.exp_settings)
        self.r4_res = None
        self.cmaq_res = None
        # TODO: update load_data_model() function to laod best performance models
        self.dataset_bundles, self.model_pools, self.max_length, self.model_load_times, self.data_load_times = self._load_data_model()

    def hard_voting(self, predictions):
        votes = Counter(predictions)
        winner = votes.most_common(1)[0][0]
        return winner

    def _load_data_model(self):
        start_time = time.time()
        dataset_bundles = []
        model_pools = []
        model_load_times = []
        data_load_times = []

        for i in tqdm(range(self.model_num)):  # 임시 코드임
            now = time.time()
            e = self.exp_settings.iloc[i]
            esv_year = e.esv_year

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
                horizon=self.horizon,
                validation_days=self.validation_days
            )

            dataset = V3Dataset(**dataset_args)
            dataset_bundles.append(dataset)
            data_load_times.append(time.time() - now)
        max_length = dataset_bundles[0].max_length
        self.r4_res = load_data(
            os.path.join(self.r4_dir, f'{self.region}_{self.pm_type}_horizon{self.horizon}_r4v3_result.pkl'))
        self.cmaq_res = load_data(
            os.path.join(self.cmaq_dir, f'{self.region}_{self.pm_type}_horizon{self.horizon}_v3cmaq_result.pkl'))
        print(f'Model & Dataset load took: {time.time() - start_time:.2f} s')
        return dataset_bundles, model_pools, max_length, model_load_times, data_load_times

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

    def _thresholding(self, array, thresholds):
        y = array.squeeze()
        y_cls = np.zeros_like(y)
        for i, threshold in enumerate(thresholds):
            y_cls[y > threshold] = i + 1
        return y_cls

    def _make_top_exps(self):
        model_types = {
            'cls_rnn': {
                'run_type': 'classification',
                'model': 'RNN'
            },
            'cls_cnn': {
                'run_type': 'classification',
                'model': 'CNN'
            },
            'reg_rnn': {
                'run_type': 'regression',
                'model': 'RNN'
            },
            'reg_cnn': {
                'run_type': 'regression',
                'model': 'CNN'
            },
        }
        results = pd.read_excel(os.path.join(self.root_dir, f'{self.region}_2021inference.xlsx'), index_col=0)
        sorted_result = results[results.val_f1 < 1].sort_values('val_f1', ascending=False)

        top_100_exp = []

        for key in model_types.keys():
            model_type = model_types[key]
            expr = f"(run_type == '{model_type['run_type']}') and (model == '{model_type['model']}') and (predict_region == '{self.region}') and (pm_type == '{self.pm_type}') and (horizon == {self.horizon})"
            res = sorted_result.query(expr)

            top_100_exp.append(res.iloc[:self.num_model_per_type])

        top_100_exp = pd.concat(top_100_exp)
        top_100_exp = top_100_exp.reset_index(drop=True)

        return top_100_exp

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
            y = y_orig_batch
        else:
            y = y_cls_batch

        x_pred = net(obs_batch, fnl_batch, num_batch, point_num)

        if is_reg:
            x_pred = x_pred.squeeze(-1) if x_pred.shape[0] == 1 else x_pred.squeeze()
            original_pred_list = x_pred.detach().clone().cpu().numpy()
            pred_list = concatenate(pred_list, x_pred.detach().clone().cpu().numpy() * scale + mean)
            y_list = concatenate(y_list, y.detach().clone().cpu().numpy())
        else:
            preds = x_pred.argmax(dim=1)
            pred_list = concatenate(pred_list, preds.detach().clone().cpu().numpy())
            original_pred_list = concatenate(original_pred_list, x_pred.detach().clone().cpu().numpy(), axis=0)
            y_list = concatenate(y_list, y.detach().clone().cpu().numpy())

        return original_pred_list, pred_list, y_list, prediction_date

    def run_v3_single(self, data, net, scale, mean, is_reg):
        _, (pred_obs, pred_fnl, pred_num, pred_y, pred_y_original, pred_y_cls, horizon_day), prediction_date = data
        original_pred = None
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
            y = pred_y_original
        else:
            y = pred_y_cls

        pred_obs = torch.cat([pred_obs, pred_obs])
        pred_fnl = torch.cat([pred_fnl, pred_fnl])
        pred_num = torch.cat([pred_num, pred_num])
        point_num = torch.cat([point_num, point_num])

        x_pred = net(pred_obs, pred_fnl, pred_num, point_num)
        x_pred = x_pred[0].unsqueeze(0)
        if is_reg:
            x_pred = x_pred.squeeze(-1) if x_pred.shape[0] == 1 else x_pred.squeeze()
            original_pred = x_pred.detach().clone().cpu().numpy()
            pred = x_pred.detach().clone().cpu().numpy() * scale + mean
        else:
            preds = x_pred.argmax(dim=1)
            pred = preds.detach().clone().cpu().numpy()
            original_pred = x_pred.detach().clone().cpu().numpy()

        y = y.detach().numpy()

        return original_pred, pred, y, prediction_date

    def _v3_validation(self, day_idx):
        f1_list = []
        for j, (model, data) in enumerate(zip(self.model_pools, self.dataset_bundles)):
            e = self.exp_settings.iloc[j]
            is_reg = True if e.run_type == 'regression' else False

            test_orig_pred, test_pred, test_label, prediction_date = self.run_v3_batch(data[day_idx], model, data.scale,
                                                                                       data.mean, is_reg)

            if is_reg:
                test_pred_score = self._thresholding(test_pred, data.threshold_dict[self.pm_type])
                test_label_score = self._thresholding(test_label, data.threshold_dict[self.pm_type])
            else:
                test_pred_score = test_pred
                test_label_score = test_label

            test_score = self._evaluation(test_label_score, test_pred_score)

            f1_list.append(test_score['f1'])

        model_len = len(f1_list)

        return f1_list, model_len

    def _handle_r4(self, day_idx, retrieve_key=None, valid=True):
        r4_res_day = self.r4_res[day_idx]
        if valid:
            return list(r4_res_day['valid_result'].values()), list(r4_res_day['valid_result'].keys())
        else:
            assert retrieve_key is not None, 'must add retrieve_keys'
            if 'clf' in retrieve_key:
                return int(r4_res_day['test_result'][retrieve_key]['label'])
            else:
                return float(r4_res_day['test_result'][retrieve_key]['pred_score'])

    def _handle_cmaq(self, day_idx, valid=True):
        cmaq_res_day = self.cmaq_res[day_idx]
        if valid:
            return cmaq_res_day['valid_result']
        else:
            return float(cmaq_res_day['test_result']['pred_score'])

    def _v3_test(self, argsorts, idx_to_key, day_idx, model_len):
        # ensemble model
        ensemble_label = None
        ensemble_pred = None
        if argsorts[0] > self.model_num - 1:
            data = self.dataset_bundles[0]
        else:
            data = self.dataset_bundles[argsorts[0]]
        for arg_idx in argsorts:
            if arg_idx < model_len:  # r5 models
                data = self.dataset_bundles[arg_idx]
                model = self.model_pools[arg_idx]
                e = self.exp_settings.iloc[arg_idx]

                is_reg = True if e.run_type == 'regression' else False
                test_orig_pred, test_pred, test_label, prediction_date = self.run_v3_single(data[day_idx], model, data.scale,
                                                                                       data.mean, is_reg)

                if is_reg:
                    test_pred_score = self._thresholding(test_pred, data.threshold_dict[self.pm_type])
                    test_label_score = self._thresholding(test_label, data.threshold_dict[self.pm_type])
                else:
                    test_pred_score = test_pred
                    test_label_score = test_label

                if len(test_pred_score.shape) == 0:
                    test_pred_score = np.array([int(test_pred_score)])
                    test_label_score = np.array([int(test_label_score)])

                ensemble_pred = concatenate(ensemble_pred, test_pred_score)
                ensemble_label = concatenate(ensemble_label, test_label_score)
            elif arg_idx == model_len + 4 and self.add_cmaq_model:
                # cmaq
                score = self._handle_cmaq(day_idx, valid=False)
                test_pred_score = self._thresholding(np.array([score]), data.threshold_dict[self.pm_type])
                ensemble_pred = concatenate(ensemble_pred, np.array([test_pred_score]))
            else:  # r4
                score = self._handle_r4(day_idx, idx_to_key[f'{arg_idx}'], valid=False)
                if type(score) is float:
                    test_pred_score = self._thresholding(np.array([score]), data.threshold_dict[self.pm_type])
                else:
                    test_pred_score = score
                ensemble_pred = concatenate(ensemble_pred, np.array([test_pred_score]))

        ensemble_label = int(ensemble_label[0])
        ensembled_prediction = self.hard_voting(ensemble_pred)

        # best model
        if argsorts[0] > self.model_num - 1:
            if argsorts[0] == model_len + 4:  # cmaq
                data = self.dataset_bundles[0]
                score = self._handle_cmaq(day_idx, valid=False)
                test_pred_score = self._thresholding(np.array([score]), data.threshold_dict[self.pm_type])
            else:  # r4 model
                score = self._handle_r4(day_idx, idx_to_key[f'{argsorts[0]}'], valid=False)
                if type(score) is float:
                    test_pred_score = self._thresholding(np.array([score]), data.threshold_dict[self.pm_type])
                else:
                    test_pred_score = score

        else:
            data = self.dataset_bundles[argsorts[0]]
            model = self.model_pools[argsorts[0]]
            e = self.exp_settings.iloc[argsorts[0]]

            is_reg = True if e.run_type == 'regression' else False
            test_orig_pred, test_pred, test_label, prediction_date = self.run_v3_single(data[day_idx], model, data.scale,
                                                                                   data.mean, is_reg)

            if is_reg:
                test_pred_score = self._thresholding(test_pred, data.threshold_dict[self.pm_type])
            else:
                test_pred_score = test_pred

            if len(test_pred_score.shape) == 0:
                test_pred_score = np.array([int(test_pred_score)])

        return ensembled_prediction, test_pred_score, ensemble_label

    def run_v3(self, top_k=9):
        ensemble_prediction_ls = []
        single_prediction_ls = []
        label_ls = []

        validation_times = []
        inference_times = []
        total_times = []

        for i in tqdm(range(self.max_length)):
            now = time.time()
            ####### validation #######
            f1_list, model_len = self._v3_validation(i)

            # r4 models
            if self.add_r4_models:
                r4_f1_list, key_list = self._handle_r4(i, valid=True)
                idx_to_key = {}
                for key_i, key in enumerate(key_list):
                    idx_to_key[f'{model_len + key_i}'] = key
                f1_list = np.concatenate([f1_list, r4_f1_list])
            # cmaq models
            if self.add_cmaq_model:
                cmaq_f1 = self._handle_cmaq(i, valid=True)
                f1_list = np.concatenate([f1_list, [cmaq_f1]])

            validation_times.append(time.time() - now)
            test_now = time.time()

            ####### test #######
            if np.mean(f1_list) <= 0:
                argsorts = np.random.permutation(len(f1_list))
            else:
                argsorts = np.argsort(f1_list)

            argsorts = argsorts[:top_k]

            ensembled_pred, single_pred, label = self._v3_test(argsorts, idx_to_key=idx_to_key, day_idx=i, model_len=model_len)

            ensemble_prediction_ls.append(ensembled_pred)
            single_prediction_ls.append(single_pred)
            label_ls.append(label)

            inference_times.append(time.time() - test_now)
            total_times.append(time.time() - now)

        return ensemble_prediction_ls, single_prediction_ls, label_ls, validation_times, inference_times, total_times


def main():

    pass

# if __name__ == "__main__":
