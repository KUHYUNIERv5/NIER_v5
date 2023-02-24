import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from . import load_data, save_data, set_random_seed
import torch
from sklearn.metrics import confusion_matrix

class Cmaq_calculate():

    def __init__(self, predict_location_id, data_path='/workspace/local/src/datagen/ver_4th/db_save/',
                 result_class=4, seed=999, original_idx_list : list=[], start_date=20190101, until_date=20211231, ):
        """
        CMAQ 모델 결과 비교를 위한 class
        :param predict_location_id: 예측하려는 지역 id (R4_59~R4_77)
        :param predict_pm: 예측값 종류 (PM10, PM25)
        :param shuffle: depricated (False로 고정)
        :param sampling: oversampling or normal (default: 'normal')
        :param data_path: root data path
        :param data_type: train or test set
        :param pca_dim: PCA 차원 수
        :param lag: 예측에 사용할 lag 길이
        :param numeric_type: 예측장 정보 종류 (WRF, CMAQ, Numeric(WRF+CMAQ))
        :param numeric_data_handling: mean 이면 하루 평균 값, single 이면 15시의 데이터 포인트
        :param horizon: 예측하려는 horizon
        :param max_lag: 최대 lag 길이 (3일 ~ 1일)
        :param max_horizon: 최대 horizon 길이 (6으로 고정)
        :param numeric_scenario: 실험 시나리오 (0: 기존 세팅(예측하려는 날 당일의 WRF), 그 외: 1일부터 d일 까지의 WRF, CMAQ 정보 사용)
        :param timepoint_day: 하루에 수집되는 데이터 포인트 수 (default: 4)
        :param interval: 예측에 사용될 interval
        :param seed: random seed
        :param serial_y: sequential 예측 여부 (True면 horizon 레이블 값, False면 해당 horizon의 레이블 반환)
        """

        self.predict_location_id = predict_location_id
        self.data_path = data_path
        self.result_class = result_class
        self.seed = seed
        self.original_idx_list = original_idx_list
        self.test_start = self._convert_date(start_date)
        self.test_end = self._convert_date(until_date)
        self.threshold_dict = dict(
            PM10=[30, 80, 150],
            PM25=[15, 35, 75]
        )
        self.two_class_threshod_dict = dict(
            PM10 = [80],
            PM25 = [35]
        )

        self.__read_data__()

        set_random_seed(self.seed)

    def _convert_date(self, date):
        convert_date = pd.to_datetime(str(date), format='%Y%m%d', errors='raise')
        return convert_date

    def __read_data__(self):
        cmaq_data = load_data(os.path.join(self.data_path, 'cmaq_r4_um.pkl'))
        obs_data = load_data(os.path.join(self.data_path, 'obs_r4_um.pkl'))
        obs_data = obs_data[['REGION_CODE', 'RAW_DATE', 'RAW_TIME', 'PM10', 'PM25']]
        obs_date_info = pd.to_datetime(obs_data['RAW_DATE'].astype(str), format='%Y%m%d', errors='raise')
        obs_data['RAW_DATE_INFO'] = obs_date_info
        obs_data = obs_data[obs_data['REGION_CODE'] == self.predict_location_id]
        obs_data = obs_data[(obs_data['RAW_DATE_INFO'] >= self.test_start) &
                            (obs_data['RAW_DATE_INFO'] <= self.test_end)]
        obs_data = obs_data.iloc[self.original_idx_list - 1]
        self.cmaq_start = obs_data['RAW_DATE_INFO'].iloc[0]
        self.cmaq_end = obs_data['RAW_DATE_INFO'].iloc[-1]
        obs_data = obs_data.set_index(['RAW_DATE', 'RAW_DATE_INFO', 'RAW_TIME', ])

        cmaq_data = cmaq_data.iloc[:, :6]
        cmaq_date_info = pd.to_datetime(cmaq_data['RAW_DATE'].astype(str), format='%Y%m%d', errors='raise')
        cmaq_data_cal = cmaq_date_info + pd.to_timedelta(cmaq_data['RAW_FDAY'], unit='d')
        cmaq_data['RESULT_DATE'] = cmaq_data_cal
        cmaq_data = cmaq_data[(cmaq_data['REGION_CODE'] == self.predict_location_id)
                              & (cmaq_data['RAW_FDAY'] > 2) & (cmaq_data['RAW_FDAY'] < 8)
                              & (cmaq_data['RESULT_DATE'] >= self.cmaq_start)
                              & (cmaq_data['RESULT_DATE'] <= self.cmaq_end)
                              & (cmaq_data['RAW_TIME'] == '15')]
        self.cmaq_data = cmaq_data

        self.cmaq_dict = {horizon : {'PM10':None, 'PM25':None } for horizon in range(3,7)}
        self.obs_dict = {'PM10':None, 'PM25':None }
        for pm_type in ['PM10', 'PM25']:
            self.obs_dict[pm_type] = obs_data[pm_type]
            for horizon in range(3, 7):
                if pm_type == 'PM10':
                    predict_pm = 'F_PM10'
                else:
                    predict_pm = 'F_PM2_5'
                cmaq_horizon = cmaq_data[cmaq_data['RAW_FDAY'] == horizon]
                cmaq_horizon = cmaq_horizon.set_index(['RESULT_DATE', 'RAW_TIME', 'RAW_FDAY'])
                self.cmaq_dict[horizon][pm_type] = cmaq_horizon[predict_pm]

        self.__thresholding__()
        self.__return_eval__()


    def _evaluation(self, y_list, pred_list):

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

    def __thresholding__(self):
        self.eval_dict = {horizon :{'PM10':[], 'PM25':[],} for horizon in range(3,7)}
        for pm_type in ['PM10', 'PM25']:
            y = self.obs_dict[pm_type].to_numpy().squeeze()
            original_y = y
            y_cls = np.zeros_like(original_y)

            if self.result_class == 4:
                for i, threshold in enumerate(self.threshold_dict[pm_type]):
                    y_cls[original_y > threshold] = i + 1
            else:
                for i, threshold in enumerate(self.two_class_threshod_dict[pm_type]):
                    y_cls[original_y > threshold] = i + 1

            for h in range(3,7):
                original_cmaq = self.cmaq_dict[h][pm_type].to_numpy().squeeze()
                cmaq_cls = np.zeros_like(original_cmaq)
                if self.result_class == 4:
                    for i, threshold in enumerate(self.threshold_dict[pm_type]):
                        cmaq_cls[original_cmaq > threshold] = i + 1
                    self.eval_dict[h][pm_type].append(y_cls)
                    self.eval_dict[h][pm_type].append(cmaq_cls)
                else:
                    for i, threshold in enumerate(self.two_class_threshod_dict[pm_type]):
                        cmaq_cls[original_cmaq > threshold] = i + 1
                    self.eval_dict[h][pm_type].append(y_cls)
                    self.eval_dict[h][pm_type].append(cmaq_cls)

    def __return_eval__(self):
        result_list = []
        for pm_type in ['PM10', 'PM25']:
            result_dict = {horizon :None for horizon in range(3,7)}
            for horizon in range(3,7):
                y_cls = self.eval_dict[horizon][pm_type][0]
                cmaq_cls = self.eval_dict[horizon][pm_type][1]
                val_score = self._evaluation(y_cls, cmaq_cls)
                result_dict[horizon] = val_score
            pm_df = pd.DataFrame(result_dict).T
            pm_df['region'] = self.predict_location_id
            pm_df['pm'] = pm_type
            pm_df = pm_df.reset_index().rename(columns={"index": "horizon"})
            pm_df = pm_df[['region', 'pm', 'horizon', 'accuracy', 'hit', 'pod', 'far', 'f1',]]
            result_list.append(pm_df)
        self.result_df = pd.concat(result_list, axis=0)
        self.result_df = self.result_df.reset_index(drop=True)
