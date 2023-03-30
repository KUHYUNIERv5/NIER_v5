#!/usr/bin/env python
# @Created   : 2022/10/21 2:27 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : dataloader_dev2.py
# @Software  : PyCharm

import os

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import Dataset, DataLoader

from ..utils import load_data


class NIERDataset(Dataset):

    def __init__(self, predict_location_id, predict_pm, shuffle=False, sampling='normal', data_path='../../dataset/d5',
                 data_type='train', pca_dim=None, lag=1, numeric_type='numeric', numeric_data_handling='single',
                 horizon=4, max_lag=7, max_horizon=6, numeric_scenario=1, timepoint_day=4, interval=1, seed=999,
                 serial_y=False, flatten=True, start_date=20170301, until_date=20220228, co2_load=False, rm_region=None, exp_name=None):
        """
        NIER Dataset 생성을 위한 class
        :param predict_location_id: 예측하려는 지역 id (R4_59~R4_77)
        :param predict_pm: 예측값 종류 (PM10, PM25)
        :param shuffle: depricated (False로 고정)
        :param sampling: oversampling or normal (default: 'normal')
        :param data_path: root data path
        :param data_type: train or test set
        :param pca_dim: PCA 차원 수
        :param lag: 예측에 사용할 lag 길이
        :param numeric_type: 예측장 정보 종류 (WRF, CMAQ, Numeric(WRF+CMAQ))
        :param numeric_data_handling: mean 이면 하루 평균 값, single 이면 15시의 데이터 포인트, normal 이면 모든 포인트(하루 4포인트)
        :param horizon: 예측하려는 horizon
        :param max_lag: 최대 lag 길이 (3일 ~ 1일)
        :param max_horizon: 최대 horizon 길이 (6으로 고정)
        :param numeric_scenario: 실험 시나리오 (0: 기존 세팅(예측하려는 날 당일의 WRF), 그 외: 1일부터 d일 까지의 WRF, CMAQ 정보 사용, 4: 1~3 + 당일)
        :param timepoint_day: 하루에 수집되는 데이터 포인트 수 (default: 4)
        :param interval: 예측에 사용될 interval
        :param seed: random seed
        :param serial_y: sequential 예측 여부 (True면 horizon 레이블 값, False면 해당 horizon의 레이블 반환)
        :rm_region: 제거 지역 그룹 개수, None 이면 제거 지역 고려 X
        """
        super(NIERDataset, self).__init__()
        if pca_dim is None:
            pca_dim = dict(
                obs=256,
                fnl=512,
                wrf=128,
                cmaq=512,
                numeric=512
            )
        assert numeric_type in ['wrf', 'cmaq', 'numeric'], f'bad numeric type: {numeric_type}'
        assert numeric_scenario in [0, 1, 2, 3, 4], f'bad scenario: {numeric_scenario}'
        self.predict_location_id = predict_location_id
        self.predict_pm = predict_pm
        self.sampling = sampling if data_type == 'train' else 'normal'
        self.data_path = data_path
        self.data_type = data_type
        self.pca_dim = pca_dim
        self.lag = lag
        self.horizon = horizon
        self.max_lag = max_lag  # 최대 lag 길이 (현재는 3, 논의 필요)
        self.max_horizon = max_horizon  # 최대 horizon 길이 (3~6 까지 예)
        self.numeric_type = numeric_type
        self.numeric_data_handling = numeric_data_handling
        self.numeric_scenario = numeric_scenario
        self.timepoint_day = timepoint_day
        self.interval = interval
        self.shuffle = shuffle
        self.seed = seed
        self.serial_y = serial_y
        self.flatten = flatten
        self.start_date = start_date
        self.until_date = until_date
        self.co2_load = co2_load
        self.rm_region = rm_region
        self.exp_name = exp_name # load data할 때 사용할 이름

        self.threshold_dict = dict(
            PM10=[30, 80, 150],
            PM25=[15, 35, 75]
        )
        
        
        # if rm_region != 0:
        #     self.rm_regions, self.rm_regions_pkl_name = self.get_rm_regions(predict_location_id, rm_region)
        # else:
        #     self.rm_regions, self.rm_regions_pkl_name = None, 'NIER_R5_data'

        self.__read_data__()

    # def get_rm_regions(self, predict_location_id, rm_region):
    #     with open(f'../NIER_R5_new/data_folder/{self.csv_name}', 'r', encoding='utf-8') as f:
    #         reader = csv.reader(f)
    #         rm_regions = None
    #         for i, line in enumerate(reader):
    #             if i > 0 and i < 9:
    #                 line = list(line)
    #                 region_num = line[0]
    #                 if region_num == predict_location_id:

    #                     if rm_region == 1:
    #                         rm_regions = line[2].split(',')
    #                     elif rm_region == 2:
    #                         rm_regions = line[2].split(',')+line[3].split(',')
    #                     elif rm_region == 3:
    #                         rm_regions = line[2].split(',')+line[3].split(',')+line[4].split(',')
    #                     rm_regions.sort()
                
    #         return rm_regions, "-".join(rm_regions)

    def __read_data__(self):
        start_year = str(self.start_date)[2:4]
        test_year = str(self.until_date)[2:4]
        
        
        whole_data = load_data(os.path.join(self.data_path, f'{self.exp_name}.pkl'))
        
        # if self.rm_region != 0:
        #     # print(self.rm_regions, "are removed")
        #     whole_data = load_data(os.path.join(self.data_path, f'{self.rm_regions_pkl_name}.pkl'))
        # elif self.rm_region == 0: # 제거 지역 고려 X
        #     # print("Using all regions")
        #     whole_data = load_data(os.path.join(self.data_path, f'NIER_R5_data.pkl'))

        self.y_ = whole_data['obs'][self.predict_pm][f'{self.data_type}_y'][self.predict_location_id]
        self.mean, self.scale = whole_data['obs'][self.predict_pm]['mean'], whole_data['obs'][self.predict_pm]['scale']

        self.obs_X = whole_data['obs']['X'][self.predict_location_id][f"pca_{self.pca_dim['obs']}"][self.data_type]
        self.fnl_X = whole_data['fnl']['X'][f"pca_{self.pca_dim['fnl']}"][self.data_type]
        self.dec_X = whole_data[self.numeric_type]['X'][f"pca_{self.pca_dim[self.numeric_type]}"][
            self.data_type].reset_index()
        self.dec_X = self.dec_X.set_index(['RAW_DATE'])

        if self.co2_load:
            co2_data = load_data(os.path.join(self.data_path, f'co2_{start_year}_to_{test_year}.pkl'))
            co2_X = co2_data[self.data_type][self.predict_location_id]
            self.obs_X = pd.concat([self.obs_X, co2_X], axis=1)

        self.max_length = (len(self.obs_X) - 3 - 4 * (
                self.max_lag + self.max_horizon)) // 4 + 1  # (len(self.obs_X) - 4 * ((self.max_lag - 1) + (self.max_horizon))) // 4
        self.idx_list = np.arange(self.max_length)
        self.original_idx_list = self.idx_list * self.timepoint_day + 3 + self.timepoint_day * self.max_lag
        self.__thresholding__()

        if self.shuffle:
            self.idx_list = np.random.permutation(self.idx_list)

        if self.sampling == 'oversampling' and self.data_type == 'train':
            self.__oversampling__()

    def __len__(self):
        return len(self.idx_list)

    def __thresholding__(self):
        y = self.y_.to_numpy().squeeze()
        self.original_y = y * self.scale + self.mean
        self.y_cls = np.zeros_like(self.original_y)
        for i, threshold in enumerate(self.threshold_dict[self.predict_pm]):
            self.y_cls[self.original_y > threshold] = i + 1

        # y = y[self.original_idx_list - 1 + self.horizon * 4]
        # original_y = self.original_y[self.original_idx_list - 1 + self.horizon * 4]
        # y_cls = np.zeros_like(original_y)
        # for i, threshold in enumerate(self.threshold_dict[self.predict_pm]):
        #     y_cls[original_y > threshold] = i + 1
        # self.y_cls = y_cls

    def __oversampling__(self):
        oversampler = RandomOverSampler(sampling_strategy='auto', random_state=self.seed)
        idx_lists = self.original_idx_list - 1 + self.horizon * 4
        dummy_X = np.random.randn(len(self.y_cls[idx_lists]), 2)
        _, _ = oversampler.fit_resample(dummy_X, self.y_cls[idx_lists])
        # print(self.original_y.shape, self.y_cls.shape, self.y_cls[idx_lists].shape)
        self.idx_list = oversampler.sample_indices_

    def __getitem__(self, item):
        horizon_day = torch.tensor([0.]).float()
        if torch.is_tensor(item):
            item = item.tolist()
        index = self.idx_list[item]

        original_idx = self.original_idx_list[
            index]  # index * self.timepoint_day + 3 + self.timepoint_day * self.max_lag

        obs_window = self.obs_X[original_idx - self.timepoint_day * self.lag:original_idx]
        fnl_window = self.fnl_X[original_idx - self.timepoint_day * self.lag:original_idx - 2]

        prediction_date = self.obs_X.index[original_idx - 1]
        num_window = self.dec_X.loc[prediction_date[0]]
        if self.numeric_scenario == 0:
            num_window = num_window[num_window.RAW_FDAY == self.horizon]
        elif self.numeric_scenario == 4 and self.horizon > 3:
            horizon_day = num_window[num_window.RAW_FDAY == self.horizon]
            num_window = num_window[num_window.RAW_FDAY.between(1, 3)]
            # point_window = pd.concat([num_window, horizon_day], axis=0)

            horizon_day = horizon_day[horizon_day.RAW_TIME == '15'].drop(['RAW_TIME', 'RAW_FDAY'], axis=1)
            # print(type(horizon_day), horizon_day.shape)
            horizon_day = torch.from_numpy(horizon_day.to_numpy()).squeeze().float()
        else:
            num_window = num_window[num_window.RAW_FDAY.between(1, self.numeric_scenario)]

        if self.numeric_data_handling == 'mean':
            num_window = num_window.drop(['RAW_TIME'], axis=1).groupby('RAW_FDAY').mean()
        elif self.numeric_data_handling == 'single':  # single
            num_window = num_window[num_window.RAW_TIME == '15'].drop(['RAW_TIME', 'RAW_FDAY'], axis=1)
        else:
            num_window = num_window.drop(['RAW_TIME', 'RAW_FDAY'], axis=1)

        if self.serial_y:
            y_window = self.y_.iloc[original_idx + 3:original_idx + 6 * 4][::4]
            y_original = self.original_y[original_idx + 3:original_idx + 6 * 4][::4]
            y_cls = self.y_cls[original_idx + 3:original_idx + 6 * 4][::4]

            y_window = y_window.to_numpy()

            y_window = torch.tensor(y_window).float().squeeze()
            y_original = torch.tensor(y_original).float().squeeze()
            y_cls = torch.tensor(y_cls).float().squeeze()

            # y_window = self.y[index - self.horizon:index + 1]
        else:
            y_window = self.y_.iloc[original_idx - 1 + self.horizon * 4]
            y_original = self.original_y[original_idx - 1 + self.horizon * 4]
            y_cls = self.y_cls[original_idx - 1 + self.horizon * 4]

            y_window = torch.tensor([y_window]).float().squeeze()
            y_original = torch.tensor([y_original]).float().squeeze()
            y_cls = torch.tensor([y_cls]).float().squeeze()

            # y_window = self.y[index]

        # print('test', obs_window, y_window)

        obs_window = torch.from_numpy(obs_window.to_numpy()).float()
        fnl_window = torch.from_numpy(fnl_window.to_numpy()).float()
        num_window = torch.from_numpy(num_window.to_numpy()).float()

        # else:
        #     horizon_day = torch.tensor([0])

        # if len(num_window.shape) > 1 and self.flatten:
        #     num_window = torch.flatten(num_window)

        # print(y_window, y_original, y_cls)

        return obs_window, fnl_window, num_window, y_window, y_original, y_cls, horizon_day


def get_dataloader(dataset_args, batch_size=64, train_shuffle=True, test_shuffle=False, num_workers=1):
    trainset = NIERDataset(**dataset_args)
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers)
    dataset_args['data_type'] = 'test'
    dataset_args['sampling'] = 'normal'
    testset = NIERDataset(**dataset_args)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=test_shuffle, num_workers=num_workers)

    return train_loader, test_loader
