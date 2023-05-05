#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2023/04/13 10:30 AM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : dataloader_v3.py
# @Software  : PyCharm

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils import load_data, save_data


class V3Dataset(Dataset):
    def __init__(self,
                 predict_region,
                 pm_type,
                 rm_region,
                 representative_region,
                 period_version,
                 data_path,
                 lag: int = 1,
                 max_lag: int = 7,
                 horizon: int = 4,
                 validation_days: int = 14,
                 inference_type: int = 2022
                 ):
        super().__init__()
        assert pm_type in ['PM10', 'PM25'], f"Unknown pm type: {pm_type}"
        assert rm_region in [i for i in range(4)], f"bad remove region code: {rm_region}"
        assert lag in [i + 1 for i in range(max_lag)], f"Bad lag: {lag}"

        self.threshold_dict = dict(
            PM10=[30, 80, 150],
            PM25=[15, 35, 75]
        )

        self.max_horizon = 6
        self.timepoint_day = 4
        self.data_type = 'test'
        self.numeric_type = 'numeric'

        self.max_lag = max_lag
        self.validation_days = validation_days
        self.predict_region = predict_region
        self.pm_type = pm_type
        self.rm_region = rm_region
        self.period_version = period_version
        self.data_path = data_path
        self.lag = lag
        self.horizon = horizon
        self.inference_type = inference_type

        self.file_name = os.path.join(predict_region,
                                      f'{predict_region}_{representative_region}_period_{period_version}_rmgroup_{rm_region}_v3')
        self._read_data()

    def _read_data(self):
        v3_file = os.path.join(self.data_path, f'{self.file_name}.pkl')
        try:
            whole_data = load_data(v3_file)
        except:
            print("No v3 file exists. Please generate v3 dataset first (run {source}/make_v3_dataset.py)")

        self.y_ = whole_data[self.pm_type]['y']
        self.mean, self.scale = whole_data[self.pm_type]['mean'], whole_data[self.pm_type]['scale']

        self.obs_X = whole_data['obs']
        self.fnl_X = whole_data['fnl']
        self.dec_X = whole_data['num'].reset_index()
        self.dec_X = self.dec_X.set_index(['RAW_DATE'])

        self.max_length = len(self.obs_X) // 4 - (self.max_lag + self.validation_days + self.max_horizon * 2)
        #         self.max_length = (len(self.obs_X) - (
        #                 self.max_lag + self.validation_days + self.max_horizon) * self.timepoint_day) // self.timepoint_day + 1
        idx_list = np.arange(self.max_length)
        self.original_idx_list = idx_list * self.timepoint_day + 3 + self.timepoint_day * (
                self.max_lag + self.validation_days + self.max_horizon)
        self._thresholding()

    def _thresholding(self):
        y = self.y_.to_numpy().squeeze()
        self.original_y = y * self.scale + self.mean
        self.y_cls = np.zeros_like(self.original_y)
        for i, threshold in enumerate(self.threshold_dict[self.pm_type]):
            self.y_cls[self.original_y > threshold] = i + 1

    def _generate_validation(self, original_idx):
        obs_batch = []
        fnl_batch = []
        num_batch = []
        horizon_day_batch = []

        y_idxs = []

        for day in range(self.validation_days):
            start_diff = (self.validation_days - day) + self.horizon + self.lag - 1
            start_idx = original_idx - self.timepoint_day * start_diff
            end_diff = self.validation_days - day + self.horizon - 1
            end_idx = original_idx - self.timepoint_day * end_diff
            y_idx = end_idx + self.timepoint_day * self.horizon - 1
            y_idxs.append(y_idx)

            obs_window = self.obs_X.iloc[start_idx:end_idx]
            fnl_window = self.fnl_X.iloc[start_idx:end_idx - 2]
            val_pred_date = self.obs_X.index[end_idx - 1]
            num_window = self.dec_X.loc[val_pred_date[0]]

            horizon_day = torch.Tensor([0.]).float()

            if self.horizon > 3:
                horizon_day = num_window[num_window.RAW_FDAY == self.horizon]
                num_window = num_window[num_window.RAW_FDAY.between(1, 3)]
                horizon_day = horizon_day[horizon_day.RAW_TIME == '15'].drop(['RAW_TIME', 'RAW_FDAY'], axis=1)
                horizon_day = torch.from_numpy(horizon_day.to_numpy()).squeeze().float()

            else:
                num_window = num_window[num_window.RAW_FDAY.between(1, 3)]

            num_window = num_window.drop(['RAW_TIME', 'RAW_FDAY'], axis=1)

            obs_batch.append(torch.Tensor(obs_window.values))
            fnl_batch.append(torch.Tensor(fnl_window.values))
            num_batch.append(torch.Tensor(num_window.values))

            horizon_day_batch.append(torch.Tensor(horizon_day))

        obs_batch = torch.stack(obs_batch)
        fnl_batch = torch.stack(fnl_batch)
        num_batch = torch.stack(num_batch)
        horizon_day_batch = torch.stack(horizon_day_batch)

        y_batch = torch.Tensor(self.y_.iloc[y_idxs].values).float().squeeze()
        y_orig_batch = torch.Tensor(self.original_y[y_idxs]).float().squeeze()
        y_cls_batch = torch.Tensor(self.y_cls[y_idxs]).float().squeeze()

        return obs_batch, fnl_batch, num_batch, y_batch, y_orig_batch, y_cls_batch, horizon_day_batch

    def __len__(self):
        return len(self.original_idx_list)

    def __getitem__(self, item):
        horizon_day = torch.Tensor([0.]).float()
        if torch.is_tensor(item):
            item = item.tolist()
        original_idx = self.original_idx_list[item]

        prediction_date = self.obs_X.index[original_idx - 1]
        prediction_idx = original_idx - 1

        obs_batch, fnl_batch, num_batch, y_batch, y_orig_batch, y_cls_batch, horizon_day_batch = self._generate_validation(
            original_idx)

        pred_obs = self.obs_X.iloc[prediction_idx - self.timepoint_day * self.lag: prediction_idx]
        pred_fnl = self.fnl_X.iloc[prediction_idx - self.timepoint_day * self.lag: prediction_idx - 2]
        pred_num = self.dec_X.loc[prediction_date[0]]

        if self.horizon > 3:
            horizon_day = pred_num[pred_num.RAW_FDAY == self.horizon]
            pred_num = pred_num[pred_num.RAW_FDAY.between(1, 3)]
            horizon_day = horizon_day[horizon_day.RAW_TIME == '15'].drop(['RAW_TIME', 'RAW_FDAY'], axis=1)
            horizon_day = torch.from_numpy(horizon_day.to_numpy()).squeeze().float()

        else:
            pred_num = pred_num[pred_num.RAW_FDAY.between(1, 3)]

        pred_num = pred_num.drop(['RAW_TIME', 'RAW_FDAY'], axis=1)

        pred_obs = torch.from_numpy(pred_obs.to_numpy()).float()
        pred_fnl = torch.from_numpy(pred_fnl.to_numpy()).float()
        pred_num = torch.from_numpy(pred_num.to_numpy()).float()
        pred_y = torch.Tensor([self.y_.iloc[prediction_idx + self.horizon * 4]])
        pred_y_original = torch.Tensor([self.original_y[prediction_idx + self.horizon * 4]])
        pred_y_cls = torch.Tensor([self.y_cls[prediction_idx + self.horizon * 4]])

        return (obs_batch, fnl_batch, num_batch, y_batch, y_orig_batch, y_cls_batch, horizon_day_batch), \
            (pred_obs, pred_fnl, pred_num, pred_y, pred_y_original, pred_y_cls, horizon_day), prediction_date


def handle_v3_data(whole_data, predict_region):
    pca_dim = dict(
        obs=256,
        fnl=512,
        wrf=128,
        cmaq=512,
        numeric=512
    )

    y_PM10 = whole_data['obs']['PM10']['test_y'][predict_region]
    mean_PM10, scale_PM10 = whole_data['obs']['PM10']['mean'], whole_data['obs']['PM10']['scale']
    y_PM25 = whole_data['obs']['PM25']['test_y'][predict_region]
    mean_PM25, scale_PM25 = whole_data['obs']['PM25']['mean'], whole_data['obs']['PM25']['scale']

    obs_X = whole_data['obs']['X'][predict_region][f"pca_{pca_dim['obs']}"]['test']
    fnl_X = whole_data['fnl']['X'][f"pca_{pca_dim['fnl']}"]['test']
    num_X = whole_data['numeric']['X'][f"pca_{pca_dim['numeric']}"]['test']

    v3_data_dict = dict(
        obs=obs_X,
        fnl=fnl_X,
        num=num_X,
    )
    v3_data_dict[f'PM10'] = {
        'y': y_PM10,
        'mean': mean_PM10,
        'scale': scale_PM10
    }
    v3_data_dict[f'PM25'] = {
        'y': y_PM25,
        'mean': mean_PM25,
        'scale': scale_PM25
    }

    return v3_data_dict


def make_v3_dataset(data_dir, region_list=np.arange(68, 78)):
    regions = [f'R4_{i}' for i in region_list]
    for region in regions:
        region_dir = os.path.join(data_dir, region)
        for file in os.listdir(region_dir):
            if not '_xai' in file and not '_v3' in file and file.endswith('.pkl'):
                name = file.split('.')[0] + '_v3'
                print(name)
                whole_data = load_data(os.path.join(region_dir, file))
                v3_data = handle_v3_data(whole_data, region)
                save_data(v3_data, region_dir, name + '.pkl')


def get_v3loader(dataset_args, num_workers=1):
    v3_dataset = V3Dataset(**dataset_args)
    v3_loader = DataLoader(dataset=v3_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    return v3_loader
