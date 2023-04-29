#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2023/04/29 3:35 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : xai_dat.py
# @Software  : PyCharm

import os
import pandas as pd
import numpy as np
import datetime
from src.utils import load_data, save_data
from abc import ABC
import torch


class XAIDataset(ABC):
    def __init__(self, region, pm_type, horizon, predict_date='20210501', data_dir='/workspace/R5_phase2/',
                 root_dir='/workspace/results/v5_phase2/'):
        super().__init__()

        self.y = None
        self.num_pca = None
        self.fnl_pca = None
        self.obs_pca = None
        self.num = None
        self.fnl = None
        self.obs = None
        self.lag = None

        self.standard_time = 15
        self.pca_dim = dict(
            obs=256,
            fnl=512,
            wrf=128,
            cmaq=512,
            numeric=512
        )
        self.threshold_dict = dict(
            PM10=[30, 80, 150],
            PM25=[15, 35, 75]
        )

        self.region = region
        self.pm_type = pm_type
        self.horizon = horizon
        self.predict_date = predict_date
        self.data_dir = data_dir
        self.root_dir = os.path.join(root_dir, region)

        csv_dir = os.path.join(root_dir, 'id_list.csv')
        exp_settings = pd.read_csv(csv_dir)
        str_expr = f"(predict_region == '{region}') and (pm_type == '{pm_type}') and (horizon == {horizon})"
        self.exp = exp_settings.query(str_expr)
        self.target_exp = self.exp.iloc[0]
        self.read_and_handle_data()

    def select_exp_setting(self, exp_id):
        expr = f"id == {exp_id}"
        self.target_exp = self.exp.query(expr)
        self.read_and_handle_data()

    def read_and_handle_data(self):
        e = self.target_exp

        representative_region = e.representative_region
        period_version = e.period_version
        rm_region = e.rm_region
        self.lag = e.lag

        file_name = os.path.join(self.region,
                                 f'{self.region}_{representative_region}_period_{period_version}_rmgroup_{rm_region}')

        data_file = os.path.join(self.data_dir, f'{file_name}.pkl')
        data = load_data(data_file)

        # handle first
        obs = data['obs'].reset_index()
        datetime_str = obs.RAW_DATE.astype(str) + obs.RAW_TIME
        obs['datetime'] = pd.to_datetime(datetime_str, format='%Y%m%d%H')
        self.obs = obs.drop(['RAW_DATE', 'RAW_TIME'], axis=1)

        fnl = data['fnl'].reset_index()
        datetime_str = fnl.RAW_DATE.astype(str) + fnl.RAW_TIME
        fnl['datetime'] = pd.to_datetime(datetime_str, format='%Y%m%d%H')
        self.fnl = fnl.drop(['RAW_DATE', 'RAW_TIME'], axis=1)

        num = data['num'].reset_index()
        datetime_str = num.RAW_DATE.astype(str) + num.RAW_TIME
        num['datetime'] = pd.to_datetime(datetime_str, format='%Y%m%d%H')
        self.num = num.drop(['RAW_DATE', 'RAW_TIME'], axis=1)

        self.obs_pca = data['obs_pca']
        self.fnl_pca = data['fnl_pca']
        self.num_pca = data['num_pca']
        self.scale = data[self.pm_type]['scale']
        self.mean = data[self.pm_type]['mean']

        y = data[self.pm_type]['y']
        y = y.reset_index()
        datetime_str = y.RAW_DATE.astype(str) + y.RAW_TIME
        y['datetime'] = pd.to_datetime(datetime_str, format='%Y%m%d%H')
        self.y = y.drop(['RAW_DATE', 'RAW_TIME'], axis=1)

    def select_date(self, predict_date='20210101'):
        target_date = pd.to_datetime(predict_date)
        target_date = target_date + pd.Timedelta(hours=self.standard_time)
        start_date = target_date - pd.Timedelta(days=self.lag) + pd.Timedelta(hours=self.standard_time)

        mask = ((self.obs['datetime'] >= start_date) & (self.obs['datetime'] <= target_date))
        assert mask.sum() is 0, "해당되는 datetime이 데이터셋에 존재하지 않습니다. 다른 date를 선택해주세요."
        num_mask = self.num['datetime'] == target_date

        obs_X = self.obs.loc[mask]
        fnl_X = self.fnl.loc[mask]
        fnl_X = fnl_X.iloc[:-2]

        num_X = self.num.loc[num_mask]

        if self.horizon > 3:
            horizon_day = num_X[num_X.RAW_FDAY == self.horizon]
            num_X = num_X[num_X.RAW_FDAY.between(1, 3)]
            horizon_day = horizon_day[horizon_day.datetime.dt.hour == 15]
            horizon_day = torch.from_numpy(
                horizon_day.drop(['RAW_FDAY', 'datetime'], axis=1).to_numpy()).squeeze().float()

        else:
            num_X = num_X[num_X.RAW_FDAY.between(1, 4)]

        num_X = num_X.drop(['RAW_FDAY', 'datetime'], axis=1)
        obs_X = obs_X.drop(['datetime'], axis=1)
        fnl_X = fnl_X.drop(['datetime'], axis=1)

        obs_X = torch.from_numpy(obs_X.to_numpy()).float()
        fnl_X = torch.from_numpy(fnl_X.to_numpy()).float()
        num_X = torch.from_numpy(num_X.to_numpy()).float()

        pred_y = self.y[self.y.datetime == target_date + pd.Timedelta(days=self.horizon)][self.pm_type].to_numpy()
        pred_y = torch.Tensor(pred_y)
        pred_y_original = pred_y * self.scale + self.mean
        pred_y_cls = 0
        for i, threshold in enumerate(self.threshold_dict[self.pm_type]):
            if pred_y_original > threshold:
                pred_y_cls = i + 1

        pred_y_cls = torch.Tensor([pred_y_cls])

        return obs_X, fnl_X, num_X, pred_y, pred_y_original, pred_y_cls


def handle_xai_data(whole_data, region):
    pca_dim = dict(
        obs=256,
        fnl=512,
        wrf=128,
        cmaq=512,
        numeric=512
    )

    y_PM10 = pd.concat([whole_data['obs']['PM10']['train_y'][region], whole_data['obs']['PM10']['test_y'][region]])
    y_PM25 = pd.concat([whole_data['obs']['PM25']['train_y'][region], whole_data['obs']['PM25']['test_y'][region]])
    mean_PM10, scale_PM10 = whole_data['obs']['PM10']['mean'], whole_data['obs']['PM10']['scale']
    mean_PM25, scale_PM25 = whole_data['obs']['PM25']['mean'], whole_data['obs']['PM25']['scale']

    obs_all = pd.concat([whole_data['obs']['X'][region][f"pca_{pca_dim['obs']}"]['train'],
                         whole_data['obs']['X'][region][f"pca_{pca_dim['obs']}"]['test']])
    obs_pca = whole_data['obs']['pca'][region]
    fnl_all = pd.concat([whole_data['fnl']['X'][f"pca_{pca_dim['fnl']}"]['train'],
                         whole_data['fnl']['X'][f"pca_{pca_dim['fnl']}"]['test']])
    fnl_pca = whole_data['fnl']['pca']
    num_all = pd.concat([whole_data['numeric']['X'][f"pca_{pca_dim['numeric']}"]['train'].reset_index(),
                         whole_data['numeric']['X'][f"pca_{pca_dim['numeric']}"]['test'].reset_index()])
    num_all = num_all.set_index(['RAW_DATE'])
    num_pca = whole_data['numeric']['pca']

    save_dicts = dict(
        obs=obs_all,
        fnl=fnl_all,
        num=num_all,
        obs_pca=obs_pca,
        fnl_pca=fnl_pca,
        num_pca=num_pca
    )

    save_dicts[f'PM10'] = {
        'y': y_PM10,
        'mean': mean_PM10,
        'scale': scale_PM10
    }
    save_dicts[f'PM25'] = {
        'y': y_PM25,
        'mean': mean_PM25,
        'scale': scale_PM25
    }

    return save_dicts


def make_xai_dataset(data_dir):
    regions = [f'R4_{i}' for i in np.arange(68, 78)]
    for region in regions:
        region_dir = os.path.join(data_dir, region)
        for file in os.listdir(region_dir):
            if not '_xai' in file and not '_v3' in file and file.endswith('.pkl'):
                name = file.split('.')[0] + '_xai'
                print(name)
                whole_data = load_data(os.path.join(region_dir, file))
                xai_data = handle_xai_data(whole_data, region)
                save_data(xai_data, region_dir, name + '.pkl')
