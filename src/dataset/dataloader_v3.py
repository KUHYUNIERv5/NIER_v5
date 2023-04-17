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

from src.utils import load_data


class V3Dataset(Dataset):
    def __init__(self,
                 predict_region,
                 predict_pm,
                 rm_region,
                 representative_region,
                 period_version,
                 data_path,
                 esv_year,
                 pca_dim=None,
                 lag: int = 1,
                 max_lag: int = 7,
                 horizon: int = 4,
                 pm_type: str = 'pm10',
                 validation_days: int = 14
                 ):
        super().__init__()

        if pca_dim is None:
            pca_dim = dict(
                obs=256,
                fnl=512,
                wrf=128,
                cmaq=512,
                numeric=512
            )

        self.pca_dim = pca_dim

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
        self.predict_pm = predict_pm
        self.rm_region = rm_region
        self.period_version = period_version
        self.data_path = data_path
        self.esv_year = esv_year
        self.lag = lag
        self.horizon = horizon
        self.pm_type = pm_type

        self.file_name = os.path.join(predict_region,
                                      f'{predict_region}_{representative_region}_period_{period_version}_rmgroup_{rm_region}')


    def _read_data(self):
        whole_data = load_data(os.path.join(self.data_path, f'{self.file_name}.pkl'))

        self.y_ = whole_data['obs'][self.pm_type][f'{self.data_type}_y'][self.predict_region]
        self.mean, self.scale = whole_data['obs'][self.pm_type]['mean'], whole_data['obs'][self.pm_type]['scale']

        self.obs_X = whole_data['obs']['X'][self.predict_region][f"pca_{self.pca_dim['obs']}"][self.data_type]
        self.fnl_X = whole_data['fnl']['X'][f"pca_{self.pca_dim['fnl']}"][self.data_type]
        self.dec_X = whole_data[self.numeric_type]['X'][f"pca_{self.pca_dim[self.numeric_type]}"][
            self.data_type].reset_index()
        self.dec_X = self.dec_X.set_index(['RAW_DATE'])

        self.max_length = (len(self.obs_X) - (
                    (3 + self.max_lag + self.validation_days + self.max_horizon) * self.timepoint_day - 1)) // self.timepoint_day + 1
        idx_list = np.arange(self.max_length)
        self.original_idx_list = idx_list * self.timepoint_day + 3 + self.timepoint_day * (self.max_lag + self.validation_days + self.horizon)
        self._thresholding()



    def _thresholding(self):
        y = self.y_.to_numpy().squeeze()
        self.original_y = y * self.scale + self.mean
        self.y_cls = np.zeros_like(self.original_y)
        for i, threshold in enumerate(self.threshold_dict[self.pm_type]):
            self.y_cls[self.original_y > threshold] = i + 1

    def _generate_validation(self, original_idx, prediction_date, prediction_idx):
        obs_batch = []
        fnl_batch = []
        num_batch = []

        y_idxs = []

        for day in range(self.validation_days):
            start_idx = original_idx - self.timepoint_day * ((self.validation_days - day) + self.lag)
            end_idx = original_idx - self.timepoint_day * (self.validation_days - day)
            y_idx = end_idx + self.timepoint_day * self.horizon - 1
            y_idxs.append(y_idx)

            obs_window = self.obs_X.iloc[start_idx:end_idx]
            fnl_window = self.fnl_X.iloc[start_idx:end_idx]
            val_pred_date = self.obs_X.index[end_idx - 1]
            num_window = self.dec_X.loc[val_pred_date[0]]

            if self.horizon > 3:
                horizon_day = num_window[num_window.RAW_FDAY == self.horizon]
                num_window = num_window[num_window.RAW_FDAY.between(1, 3)]
                horizon_day = horizon_day[horizon_day.RAW_TIME == '15'].drop(['RAW_TIME', 'RAW_FDAY'], axis=1)
                horizon_day = torch.from_numpy(horizon_day.to_numpy()).squeeze().float()

            else:
                num_window = num_window[num_window.RAW_FDAY.between(1, 4)]

            num_window = num_window.drop(['RAW_TIME', 'RAW_FDAY'], axis=1)

            obs_batch.append(torch.tensor(obs_window.values))
            fnl_batch.append(torch.tensor(fnl_window.values))
            num_batch.append(torch.tensor(num_window.values))

        obs_batch = torch.stack(obs_batch)
        fnl_batch = torch.stack(fnl_batch)
        num_batch = torch.stack(num_batch)

        y_batch = torch.tensor(self.y_.iloc[y_idxs].values).float().squeeze()
        y_orig_batch = torch.tensor(self.original_y[y_idxs]).float().squeeze()
        y_cls_batch = torch.tensor(self.y_cls[y_idxs]).float().squeeze()

        return obs_batch, fnl_batch, num_batch, y_batch, y_orig_batch, y_cls_batch



    def __len__(self):
        print(len(self.original_idx_list))
        return len(self.original_idx_list)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        original_idx = self.original_idx_list[item]

        prediction_date = self.obs_X.index[original_idx - 1]
        prediction_idx = original_idx - 1

        obs_batch, fnl_batch, num_batch, y_batch, y_orig_batch, y_cls_batch = self._generate_validation(original_idx, prediction_date, prediction_idx)

        pred_obs = self.obs_X.iloc[prediction_idx - self.timepoint_day * self.lag: prediction_idx]
        pred_fnl = self.fnl_X.iloc[prediction_idx - self.timepoint_day * self.lag: prediction_idx]
        pred_num = self.dec_X.loc[prediction_date[0]]

        if self.horizon > 3:
            horizon_day = pred_num[pred_num.RAW_FDAY == self.horizon]
            pred_num = pred_num[pred_num.RAW_FDAY.between(1, 3)]
            horizon_day = horizon_day[horizon_day.RAW_TIME == '15'].drop(['RAW_TIME', 'RAW_FDAY'], axis=1)
            horizon_day = torch.from_numpy(horizon_day.to_numpy()).squeeze().float()

        else:
            pred_num = pred_num[pred_num.RAW_FDAY.between(1, 4)]

        pred_num = pred_num.drop(['RAW_TIME', 'RAW_FDAY'], axis=1)

        pred_obs = torch.from_numpy(pred_obs.to_numpy()).float()
        pred_fnl = torch.from_numpy(pred_fnl.to_numpy()).float()
        pred_num = torch.from_numpy(pred_num.to_numpy()).float()
        pred_y = torch.tensor([self.y_.iloc[prediction_idx + self.horizon * 4]])
        pred_y_original = torch.tensor([self.original_y[prediction_idx + self.horizon * 4]])
        pred_y_cls = torch.tensor([self.y_cls[prediction_idx + self.horizon * 4]])

        return (obs_batch, fnl_batch, num_batch, y_batch, y_orig_batch, y_cls_batch), \
            (pred_obs, pred_fnl, pred_num, pred_y, pred_y_original, pred_y_cls)

def get_dataloader(dataset_args, shuffle=False, num_workers=1):
    v3_dataset = V3Dataset(**dataset_args)
    v3_loader = DataLoader(dataset=v3_dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers)
    return v3_loader


