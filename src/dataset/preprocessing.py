#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/10/17 9:05 PM
# @Author    : Junhyung Kwon
# @Site      :
# @File      : preprocessing.py
# @Software  : PyCharm

import os

import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ..utils import region_flatten_cwdb, db_to_pkl, check_array, load_data, save_data

from abc import ABC

"""
Phase2에서 변경된 점:
1. 21년도까지 전체 데이터로 학습하는 것으로 변경됨. 그에 따라 start_date, until_date, test_date 개념은 폐기, period_version으로 대체
2. 또한 'train_periods' dict를 활용해서 p1~p4까지 최적화된 입력기간을 선정하도록 변경함. '[trainset_startdate, trainset_enddate]' 형식으로 지정함
3. 저장되는 중간 object 파일 명 역시 um_obs_R4_{start_year}_to_{test_year}_230127.pkl -> um_obs_R4_train{period_version}_test{test_period_version}.pkl 로 변경됨
    - p1: 17-21
    - p2: 18-21
    - p3: 19-21
    - p4: 21
4. test_period는 22년도 데이터 완성후에 v2가 default이며, 그전까지는 tmp 버전 사용
"""


class MakeNIERDataset(ABC):
    def __init__(self, reset_db=False, period_version='p1', test_period_version='v2', seed=999, reset_predata=False,
                 preprocess_root='../dataset/d5_phase2', root_dir="/workspace/local/src/datagen/ver_4th/db_save",
                 save_processed_data=True, run_pca=True, predict_region='R4_62', representative_region='R4_62',
                 remove_region=0, yaml_dir='', rmgroup_file='../NIER_v5/data_folder/height_region_list.csv'):
        super(MakeNIERDataset, self).__init__()
        end_date = 20211231
        if test_period_version == 'v1':
            end_date = 20201231
        train_periods = dict(
            p1=[20170301, end_date],
            p2=[20180101, end_date],
            p3=[20190101, end_date],
            p4=[20200101, end_date],
            p5=[20170101, 20191231],
            p6=[20200101, 20211231]
        )
        test_periods = dict(
            v1=[20210101, 20211231],
            v2=[20220101, 20221231],
            tmp=[20211201, 20211231]
        )
        self.yaml_dir = yaml_dir
        self.reset_predata = reset_predata
        self.reset_db = reset_db
        self.preprocess_root = preprocess_root
        self.root_dir = root_dir
        self.period_version = period_version
        self.test_period_version = test_period_version
        self.train_period = train_periods[period_version]
        self.test_period = test_periods[test_period_version]
        # self.start_date = start_date
        # self.until_date = until_date
        # self.start_year = str(self.start_date)[2:4]
        # self.test_year = str(self.until_date)[2:4]
        # self.test_date = test_date
        self.representative_region = representative_region  # 해당 지역(predict_region)의 대표권역을 의미
        self.save_processed_data = save_processed_data
        self.remove_region = remove_region
        self.seed = seed

        self.predict_region = predict_region
        self.rm_regions = self.build_rm_region(rmgroup_file, remove_region)  # list of region to remove

        self.preprocess()

        if run_pca:
            self.handle_pca()
        else:
            print("PCA is not applied")
            pass

    # predict_region -> representative_region
    def build_rm_region(self, rmgroup_file, remove_region):
        """region 별 remove_region num에 대한 region list를 만드는 함수

        Args:
            rmgroup_file (str): rmgroup_file path
            remove_region (int): number of region group to remove
            representative_region
        """
        df = pd.read_csv(rmgroup_file)
        regions = df[df.Region == self.representative_region].iloc[:, 2:2 + remove_region].values.squeeze(0).tolist()
        rm_region_list = []
        for x in regions:
            rm_region_list.extend(x.split(","))

        rm_region_list = list(set(rm_region_list))
        print("rm_region_list: ", rm_region_list)

        return rm_region_list

    def _is_predata_exists(self):
        name = f'R4_train{self.period_version}_test{self.test_period_version}.pkl'
        data_list = ['obs', 'fnl', 'wrf', 'cmaq', 'cwdb', 'ewkr']
        path_list = [os.path.join(self.preprocess_root, f'um_{data_name}_{name}') for data_name in data_list]

        return os.path.exists(path_list[0]) and os.path.exists(path_list[1]) and \
            os.path.exists(path_list[2]) and os.path.exists(path_list[3]) and \
            os.path.exists(path_list[4]) and os.path.exists(path_list[5])

    def preprocess(self):

        if self.save_processed_data and (not self._is_predata_exists() or self.reset_predata):
            obs_df, fnl_df, wrf_df, cmaq_df, cwdb_df, ewkr_df = self._select_by_date()

            obs_train, obs_test, obs_scaler = self._preprocess_df(obs_df, 'obs')  # , test_date=self.test_date
            cw_train, cw_test, cw_scaler = self._preprocess_df(cwdb_df, 'cwdb')  # , test_date=self.test_date
            ewkr_train, ewkr_test, ewkr_scaler = self._preprocess_df(ewkr_df, 'ewkr')  # , test_date=self.test_date
            fnl_train, fnl_test, fnl_scaler = self._preprocess_df(fnl_df, 'fnl')  # , test_date=self.test_date
            wrf_train, wrf_test, wrf_scaler = self._preprocess_df(wrf_df, 'wrf')  # , test_date=self.test_date
            cmaq_train, cmaq_test, cmaq_scaler = self._preprocess_df(cmaq_df, 'cmaq')  # , test_date=self.test_date

            obs = dict(
                train=obs_train,
                test=obs_test,
                scaler=obs_scaler
            )
            cw = dict(
                train=cw_train,
                test=cw_test,
                scaler=cw_scaler
            )
            ewkr = dict(
                train=ewkr_train,
                test=ewkr_test,
                scaler=ewkr_scaler
            )
            fnl = dict(
                train=fnl_train,
                test=fnl_test,
                scaler=fnl_scaler
            )
            wrf = dict(
                train=wrf_train,
                test=wrf_test,
                scaler=wrf_scaler
            )
            cmaq = dict(
                train=cmaq_train,
                test=cmaq_test,
                # train_pm=cmaq_train_pm,
                # test_pm=cmaq_test_pm,
                scaler=cmaq_scaler
            )

            self._save_preprocessed_v1(obs, cw, ewkr, fnl, wrf, cmaq)
        else:
            obs, cw, ewkr, fnl, wrf, cmaq = self._load_preprocessed_v1()

        self.obs = obs
        self.cw = cw
        self.ewkr = ewkr
        self.fnl = fnl
        self.wrf = wrf
        self.cmaq = cmaq

    def _load_preprocessed_v1(self):
        # start_year = str(self.start_date)[2:4]
        # test_year = str(self.until_date)[2:4]
        # name = 'R4_{start_year}_to_{test_year}_221018.pkl'
        name = f'R4_train{self.period_version}_test{self.test_period_version}.pkl'

        obs = load_data(os.path.join(self.preprocess_root, f'um_obs_{name}'))
        cw = load_data(os.path.join(self.preprocess_root, f'um_cw_{name}'))
        ewkr = load_data(os.path.join(self.preprocess_root, f'um_ewkr_{name}'))
        fnl = load_data(os.path.join(self.preprocess_root, f'um_fnl_{name}'))
        wrf = load_data(os.path.join(self.preprocess_root, f'um_wrf_{name}'))
        cmaq = load_data(os.path.join(self.preprocess_root, f'um_cmaq_{name}'))

        return obs, cw, ewkr, fnl, wrf, cmaq

    def _save_preprocessed_v1(self, obs, cw, ewkr, fnl, wrf, cmaq):
        # start_year = str(self.start_date)[2:4]
        # test_year = str(self.until_date)[2:4]
        name = f'R4_train{self.period_version}_test{self.test_period_version}.pkl'

        save_data(obs, self.preprocess_root, filename=f'um_obs_{name}')
        save_data(cw, self.preprocess_root, filename=f'um_cw_{name}')
        save_data(ewkr, self.preprocess_root, filename=f'um_ewkr_{name}')
        save_data(fnl, self.preprocess_root, filename=f'um_fnl_{name}')
        save_data(wrf, self.preprocess_root, filename=f'um_wrf_{name}')
        save_data(cmaq, self.preprocess_root, filename=f'um_cmaq_{name}')

    def _check_data(self, obs_df, fnl_df, wrf_df, cmaq_df):
        s_date = str(self.train_period[0])[0:4] + '-' + str(self.train_period[0])[4:6] + '-' + str(
            self.train_period[0])[6:8]
        u_date = str(self.train_period[1])[0:4] + '-' + str(self.train_period[1])[4:6] + '-' + str(
            self.train_period[1])[6:8]
        dates = pd.date_range(s_date, u_date, freq='D')
        duration = pd.to_datetime(dates.astype(str), format='%Y%m%d', errors='ignore')
        year = duration.str[0:4]
        mon = duration.str[5:7]
        day = duration.str[8:10]
        duration = year + mon + day
        period = list(duration.astype(int))

        times = ['03', '09', '15', '21']
        region_list = list(wrf_df['REGION_CODE'].unique())

        real_numeric_date = []
        real_date = []
        for region in region_list:
            for date in period:
                for time in times:
                    item1 = (region, date, time)
                    real_date.append(item1)
                    for f in range(1, 8):
                        item2 = (region, date, time, f)
                        real_numeric_date.append(item2)

        obs = obs_df.set_index(['REGION_CODE', 'RAW_DATE', 'RAW_TIME'])
        fnl = fnl_df.set_index(['REGION_CODE', 'RAW_DATE', 'RAW_TIME'])
        wrf = wrf_df.set_index(['REGION_CODE', 'RAW_DATE', 'RAW_TIME', 'RAW_FDAY'])
        cmaq = cmaq_df.set_index(['REGION_CODE', 'RAW_DATE', 'RAW_TIME', 'RAW_FDAY'])
        wrf_date = set(list(wrf.index))
        cmaq_date = set(list(cmaq.index))
        obs_date = set(list(obs.index))
        fnl_date = set(list(fnl.index))

        temp1 = [x for x in real_date if x not in obs_date]
        if len(temp1) > 0:
            print('Real Record num: ', len(real_date), 'OBS Record num: ', len(obs_date), 'FNL Record num: ',
                  len(fnl_date))
            print('OBS Missing Record: ', len(temp1))
            with open('./data_error_log/obs_missing_record.csv', 'w') as file:
                write = csv.writer(file)
                write.writerow(temp1)
        else:
            print("All OBS Record Correct")

        temp2 = [x for x in real_date if x not in fnl_date]
        if len(temp2) > 0:
            print('Real Record num: ', len(real_date), 'OBS Record num: ', len(obs_date), 'FNL Record num: ',
                  len(fnl_date))
            print('FNL Missing Record: ', len(temp2))
            with open('./data_error_log/fnl_missing_record.csv', 'w') as file:
                write = csv.writer(file)
                write.writerow(temp2)
        else:
            print("All FNL Record Correct")

        temp3 = [x for x in real_numeric_date if x not in wrf_date]
        if len(temp3) > 0:
            print('Real Numeric Record num: ', len(real_numeric_date), 'WRF Record num: ', len(wrf_date),
                  'CMAQ Record num: ', len(cmaq_date))
            print('WRF Missing Record: ', len(temp3))
            with open('./data_error_log/wrf_missing_record.csv', 'w') as file:
                write = csv.writer(file)
                write.writerow(temp3)
        else:
            print("All WRF Record Correct")

        temp4 = [x for x in real_numeric_date if x not in cmaq_date]
        if len(temp4) > 0:
            print('Real Numeric Record num: ', len(real_numeric_date), 'WRF Record num: ', len(wrf_date),
                  'CMAQ Record num: ', len(cmaq_date))
            print('CMAQ Missing Record: ', len(temp4))
            with open('./data_error_log/cmaq_missing_record.csv', 'w') as file:
                write = csv.writer(file)
                write.writerow(temp4)
        else:
            print("All CMAQ Record Correct")

    def _select_by_date(self):
        obs_df, fnl_df, wrf_df, cmaq_df, cwdb_df, ewkr_df = db_to_pkl(get_data=self.reset_db,
                                                                      root_dir=self.root_dir,
                                                                      yaml_dir=self.yaml_dir)

        ############## 날짜 튜닝 시 작업해야할 부분 ######################

        obs_df = obs_df[(obs_df['RAW_DATE'] >= self.train_period[0]) & (obs_df['RAW_DATE'] <= self.test_period[1])]
        fnl_df = fnl_df[(fnl_df['RAW_DATE'] >= self.train_period[0]) & (fnl_df['RAW_DATE'] <= self.test_period[1])]
        wrf_df = wrf_df[(wrf_df['RAW_DATE'] >= self.train_period[0]) & (wrf_df['RAW_DATE'] <= self.test_period[1])]
        cmaq_df = cmaq_df[(cmaq_df['RAW_DATE'] >= self.train_period[0]) & (cmaq_df['RAW_DATE'] <= self.test_period[1])]
        cwdb_df = cwdb_df[(cwdb_df['RAW_DATE'] >= self.train_period[0]) & (cwdb_df['RAW_DATE'] <= self.test_period[1])]
        ewkr_df = ewkr_df[(ewkr_df['RAW_DATE'] >= self.train_period[0]) & (ewkr_df['RAW_DATE'] <= self.test_period[1])]
        wrf_df = wrf_df[wrf_df['RAW_FDAY'] < 8]
        cmaq_df = cmaq_df[cmaq_df['RAW_FDAY'] < 8]

        ############################################################

        obs_df["WEEK_NO_CN"].loc[obs_df["WEEK_NO_CN"] < 6] = 0
        obs_df["WEEK_NO_CN"].loc[obs_df["WEEK_NO_CN"] >= 6] = 1
        obs_df["WEEK_NO_KR"].loc[obs_df["WEEK_NO_KR"] < 6] = 0
        obs_df["WEEK_NO_KR"].loc[obs_df["WEEK_NO_KR"] >= 6] = 1
        obs_df["ASOS_RN"].loc[obs_df["ASOS_RN"] > 0] = 1

        obs_df["RAW_MONTH"] = obs_df['RAW_DATE'].astype(str).str[4:6].astype(int)

        obs_df = obs_df[
            ['REGION_CODE', 'RAW_DATE', 'RAW_TIME', 'WEEK_NO_KR', 'WEEK_NO_CN', 'RAW_MONTH', 'ASOS_U', 'ASOS_V',
             'ASOS_WS',
             'ASOS_PA', 'ASOS_TA', 'ASOS_TD', 'ASOS_RH', 'ASOS_RN', 'PM10', 'PM25', 'O3', 'NO2', 'CO', 'SO2']]

        self._check_data(obs_df, fnl_df, wrf_df, cmaq_df)

        return obs_df, fnl_df, wrf_df, cmaq_df, cwdb_df, ewkr_df

    def _preprocess_df(self, df, df_type):  # , test_date=20210301
        assert df_type in ['obs', 'fnl', 'wrf', 'cmaq', 'cwdb', 'ewkr'], 'data_type: invalid parameter %s' % df_type
        if df_type == 'cwdb':
            df = region_flatten_cwdb(df, df_type)
            df = df.reset_index()

        # if year of test_date is same as year of start_date then train_df RAW_DATE > test_date
        # if int(str(test_date)[:4]) == int(str(self.start_date)[:4]):
        #     train_df = df[df.RAW_DATE > test_date]
        #     test_df = df[df.RAW_DATE <= test_date]
        # else:
        #     train_df = df[df.RAW_DATE < test_date]
        #     test_df = df[df.RAW_DATE >= test_date]

        # phase2 version으로 변경함
        train_df = df[(df.RAW_DATE >= self.train_period[0]) & (df.RAW_DATE <= self.train_period[1])]
        test_df = df[(df.RAW_DATE >= self.test_period[0]) & (df.RAW_DATE <= self.test_period[1])]

        if df_type == 'cmaq' or df_type == 'wrf':
            index_cols = ['RAW_DATE', 'RAW_TIME', 'RAW_FDAY']
        else:
            index_cols = ['RAW_DATE', 'RAW_TIME']
        train_df = train_df.set_index(index_cols)
        test_df = test_df.set_index(index_cols)
        train_df_region = train_df.REGION_CODE
        test_df_region = test_df.REGION_CODE

        common_cols = []

        if df_type == 'obs':
            common_cols = ['WEEK_NO_KR', 'WEEK_NO_CN', 'RAW_MONTH', 'REGION_CODE']
        else:
            common_cols = ['REGION_CODE']
        # elif df_type == 'fnl' or df_type == 'ewkr' or df_type == 'cwdb':
        #     common_cols = ['REGION_CODE']
        # elif df_type == 'numeric':  # numeric
        #     common_cols = ['REGION_CODE', 'RAW_FDAY']

        if len(common_cols) > 0:
            train_df = train_df.drop(common_cols, axis=1)
            test_df = test_df.drop(common_cols, axis=1)

        scaler = StandardScaler()
        train_df_scaled = scaler.fit_transform(train_df)
        test_df_scaled = scaler.transform(test_df)

        train_df = pd.DataFrame(train_df_scaled, index=train_df.index, columns=train_df.columns)
        test_df = pd.DataFrame(test_df_scaled, index=test_df.index, columns=test_df.columns)
        train_df['REGION_CODE'] = train_df_region
        test_df['REGION_CODE'] = test_df_region

        regions = train_df.REGION_CODE.unique().tolist()

        train_region_dict = {}
        test_region_dict = {}

        train_flat_df = []
        test_flat_df = []

        # !!!!!!!!!!!!!!!!!!!!!!!!!!! CMAQ 부분이 변경됨 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # if df_type == 'cmaq':
        #     train_pm = {}
        #     test_pm = {}
        # !!!!!!!!!!!!!!!!!!!!!!!!!!! CMAQ 부분이 변경됨 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        for i, region in enumerate(regions):
            region_id = int(region.split("_")[1])
            train_d = train_df.loc[train_df.REGION_CODE == region]
            test_d = test_df.loc[test_df.REGION_CODE == region]
            train_d = train_d.drop(['REGION_CODE'], axis=1)
            test_d = test_d.drop(['REGION_CODE'], axis=1)

            if region_id < 59:
                rename_idx = len(train_df.columns)
                for col in train_d.columns[-(rename_idx - 1):]:
                    train_d = train_d.rename(columns={col: '[' + region + ']' + col}, errors="raise")
                    test_d = test_d.rename(columns={col: '[' + region + ']' + col}, errors="raise")

                train_flat_df.append(train_d)
                test_flat_df.append(test_d)
            else:
                train_region_dict[region] = train_d
                test_region_dict[region] = test_d
            # !!!!!!!!!!!!!!!!!!!!!!!!!!! CMAQ 부분이 변경됨 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # if df_type == 'cmaq':
            #     train_pm['region'] = train_d[['F_PM10', 'F_PM2_5']]
            #     test_pm['region'] = test_d[['F_PM10', 'F_PM2_5']]
            # !!!!!!!!!!!!!!!!!!!!!!!!!!! CMAQ 부분이 변경됨 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # Concatenate chinese regions (< R4_59)
        if len(train_flat_df) > 0:
            chinese_region_train = pd.concat(train_flat_df, axis=1)
            chinese_region_test = pd.concat(test_flat_df, axis=1)

            train_region_dict['chinese_region'] = chinese_region_train
            test_region_dict['chinese_region'] = chinese_region_test

        return train_region_dict, test_region_dict, scaler
        # !!!!!!!!!!!!!!!!!!!!!!!!!!! CMAQ 부분이 변경됨 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # if df_type != 'cmaq':
        #     return train_region_dict, test_region_dict, scaler
        # else:
        #     return train_region_dict, test_region_dict, train_pm, test_pm, scaler
        # !!!!!!!!!!!!!!!!!!!!!!!!!!! CMAQ 부분이 변경됨 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def handle_pca(self):
        dataset_list = ['obs', 'fnl', 'wrf', 'cmaq', 'numeric']
        pca_latent_dims = dict(
            obs=[64, 128, 256, 512],
            fnl=[256, 512, 1024],
            wrf=[64, 128],
            cmaq=[256, 512, 1024]
        )
        df_list = dict(
            obs=self.obs,
            fnl=self.fnl,
            wrf=self.wrf,
            cmaq=self.cmaq
        )

        self.final_data = dict()

        for data_type in dataset_list:
            if data_type == 'numeric':
                self.final_data[data_type] = self._pca_fitting(df_list['cmaq'], data_type, pca_latent_dims['cmaq'])
            else:
                self.final_data[data_type] = self._pca_fitting(df_list[data_type], data_type,
                                                               pca_latent_dims[data_type])

        ####################

        # 수정 => 지역명 폴더 생성이 안되서 에러났었음.
        if not os.path.exists(os.path.join(self.preprocess_root, self.predict_region)):
            os.makedirs(os.path.join(self.preprocess_root, self.predict_region))

        # 수정 => .pkl이 두번 생겨서 save_path에 .pkl 1개 지움.
        save_path = os.path.join(self.predict_region,
                                 # f'{self.predict_region}_{self.start_date}_{self.until_date}_rmgroup_{self.remove_region}')
                                 f'{self.predict_region}_{self.representative_region}_period_{self.period_version}_rmgroup_{self.remove_region}')
        save_data(self.final_data, self.preprocess_root, f"{save_path}.pkl")

    def _pca_fitting(self, main_df, df_type, pca_latent_dim_list=None):
        if pca_latent_dim_list is None:
            pca_latent_dim_list = [64, 128, 256, 512]

        return_data = {
            'X': {},
            'pca': {}
        }

        train_flat = []
        test_flat = []

        regions = list(main_df['train'].keys())

        ########### region tuning 시 바꿔야 할 부분 #############

        for i, region in enumerate(regions):
            if region in self.rm_regions:
                print('remove region: ', region)
            else:
                train_flat.append(main_df['train'][region])
                test_flat.append(main_df['test'][region])

            # !!!!!!!!!!!!!!!!!!!!!!!!!!! CMAQ 부분이 변경됨 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #             if df_type == 'cmaq' or df_type == 'numeric':
        #                 cmaq_train_pm = []
        #                 cmaq_test_pm = []
        # !!!!!!!!!!!!!!!!!!!!!!!!!!! CMAQ 부분이 변경됨 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        #####################################################

        if df_type == 'obs':
            pm_idx = {
                'PM10': -6,
                'PM25': -5
            }

            pm10_mean, pm10_var = main_df['scaler'].mean_[pm_idx['PM10']], main_df['scaler'].scale_[pm_idx['PM10']]
            pm25_mean, pm25_var = main_df['scaler'].mean_[pm_idx['PM25']], main_df['scaler'].scale_[pm_idx['PM25']]

            return_data['PM10'] = {
                'mean': pm10_mean,
                'scale': pm10_var,
                'train_y': {},
                'test_y': {}
            }
            return_data['PM25'] = {
                'mean': pm25_mean,
                'scale': pm25_var,
                'train_y': {},
                'test_y': {}
            }

            """
            기존에는 pca를 모든 predict region에 대해서 했으나, remove region이 들어간 상태에서는 predict region마다 파일을 저장하기 때문에
            list로 저장할 필요가 없다는 판단하에 삭제 (Junhyung).
            """

            # region_list = list(main_df['train'].keys())
            # try:
            #     region_list.remove('chinese_region')
            # except ValueError as e:
            #     print(e)

            # for j, predict_region in enumerate(region_list):

            predict_region = self.predict_region

            return_data['X'][predict_region] = {}

            # if j > 0:
            #     _ = train_flat.pop()
            #     _ = test_flat.pop()

            train_flat.append(self.cw['train'][predict_region])
            test_flat.append(self.cw['test'][predict_region])

            return_data['PM10']['train_y'][predict_region] = main_df['train'][predict_region]['PM10']
            return_data['PM10']['test_y'][predict_region] = main_df['test'][predict_region]['PM10']
            return_data['PM25']['train_y'][predict_region] = main_df['train'][predict_region]['PM25']
            return_data['PM25']['test_y'][predict_region] = main_df['test'][predict_region]['PM25']

            trainset = pd.concat(train_flat, axis=1)
            testset = pd.concat(test_flat, axis=1)

            pca = PCA(svd_solver='full', random_state=self.seed)
            pca.fit(trainset)
            return_data['pca'][predict_region] = pca

            for latent_dim in pca_latent_dim_list:
                return_data['X'][predict_region][f'pca_{latent_dim}'] = {}
                train_pca = np.dot(check_array(trainset) - pca.mean_, pca.components_[:latent_dim].T)
                test_pca = np.dot(check_array(testset) - pca.mean_, pca.components_[:latent_dim].T)
                train_pca = pd.DataFrame(train_pca)
                test_pca = pd.DataFrame(test_pca)

                train_X = train_pca.set_index(trainset.index)
                test_X = test_pca.set_index(testset.index)

                return_data['X'][predict_region][f'pca_{latent_dim}']['train'] = train_X
                return_data['X'][predict_region][f'pca_{latent_dim}']['test'] = test_X

        elif df_type == 'numeric':
            regions = list(self.wrf['train'].keys())
            ########### region tuning 시 바꿔야 할 부분 #############

            for i, region in enumerate(regions):
                if region in self.rm_regions:
                    print('numeric - remove region: ', region)
                else:
                    train_flat.append(self.wrf['train'][region])
                    test_flat.append(self.wrf['test'][region])

            #####################################################

            trainset = pd.concat(train_flat, axis=1)
            testset = pd.concat(test_flat, axis=1)

            pca = PCA(svd_solver='full', random_state=self.seed)
            pca.fit(trainset)
            return_data['pca'] = pca

            for latent_dim in pca_latent_dim_list:
                return_data['X'][f'pca_{latent_dim}'] = {}
                train_pca = np.dot(check_array(trainset) - pca.mean_, pca.components_[:latent_dim].T)
                test_pca = np.dot(check_array(testset) - pca.mean_, pca.components_[:latent_dim].T)
                train_pca = pd.DataFrame(train_pca)
                test_pca = pd.DataFrame(test_pca)

                train_X = train_pca.set_index(trainset.index)
                test_X = test_pca.set_index(testset.index)

                return_data['X'][f'pca_{latent_dim}']['train'] = train_X
                return_data['X'][f'pca_{latent_dim}']['test'] = test_X

        else:
            if df_type == 'fnl':
                train_flat.append(self.ewkr['train']['chinese_region'])
                test_flat.append(self.ewkr['test']['chinese_region'])

            trainset = pd.concat(train_flat, axis=1)
            testset = pd.concat(test_flat, axis=1)

            pca = PCA(svd_solver='full', random_state=self.seed)
            pca.fit(trainset)
            return_data['pca'] = pca

            for latent_dim in pca_latent_dim_list:
                return_data['X'][f'pca_{latent_dim}'] = {}
                train_pca = np.dot(check_array(trainset) - pca.mean_, pca.components_[:latent_dim].T)
                test_pca = np.dot(check_array(testset) - pca.mean_, pca.components_[:latent_dim].T)
                train_pca = pd.DataFrame(train_pca)
                test_pca = pd.DataFrame(test_pca)

                train_X = train_pca.set_index(trainset.index)
                test_X = test_pca.set_index(testset.index)

                return_data['X'][f'pca_{latent_dim}']['train'] = train_X
                return_data['X'][f'pca_{latent_dim}']['test'] = test_X

        return return_data

    def _pca_fitting_original(self, main_df, df_type, pca_latent_dim_list=None):
        if pca_latent_dim_list is None:
            pca_latent_dim_list = [64, 128, 256, 512]

        return_data = {
            'X': {},
            'pca': {}
        }

        train_flat = []
        test_flat = []

        regions = list(main_df['train'].keys())

        ########### region tuning 시 바꿔야 할 부분 #############

        for i, region in enumerate(regions):
            if region in self.rm_regions:
                print('remove region: ', region)
            else:
                train_flat.append(main_df['train'][region])
                test_flat.append(main_df['test'][region])

            # !!!!!!!!!!!!!!!!!!!!!!!!!!! CMAQ 부분이 변경됨 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #             if df_type == 'cmaq' or df_type == 'numeric':
        #                 cmaq_train_pm = []
        #                 cmaq_test_pm = []
        # !!!!!!!!!!!!!!!!!!!!!!!!!!! CMAQ 부분이 변경됨 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        #####################################################

        if df_type == 'obs':
            pm_idx = {
                'PM10': -6,
                'PM25': -5
            }

            pm10_mean, pm10_var = main_df['scaler'].mean_[pm_idx['PM10']], main_df['scaler'].scale_[pm_idx['PM10']]
            pm25_mean, pm25_var = main_df['scaler'].mean_[pm_idx['PM25']], main_df['scaler'].scale_[pm_idx['PM25']]

            return_data['PM10'] = {
                'mean': pm10_mean,
                'scale': pm10_var,
                'train_y': {},
                'test_y': {}
            }
            return_data['PM25'] = {
                'mean': pm25_mean,
                'scale': pm25_var,
                'train_y': {},
                'test_y': {}
            }

            region_list = list(main_df['train'].keys())
            try:
                region_list.remove('chinese_region')
            except ValueError as e:
                print(e)

            for j, predict_region in enumerate(region_list):
                return_data['X'][predict_region] = {}

                if j > 0:
                    _ = train_flat.pop()
                    _ = test_flat.pop()

                train_flat.append(self.cw['train'][predict_region])
                test_flat.append(self.cw['test'][predict_region])

                return_data['PM10']['train_y'][predict_region] = main_df['train'][predict_region]['PM10']
                return_data['PM10']['test_y'][predict_region] = main_df['test'][predict_region]['PM10']
                return_data['PM25']['train_y'][predict_region] = main_df['train'][predict_region]['PM25']
                return_data['PM25']['test_y'][predict_region] = main_df['test'][predict_region]['PM25']

                trainset = pd.concat(train_flat, axis=1)
                testset = pd.concat(test_flat, axis=1)

                pca = PCA(svd_solver='full', random_state=self.seed)
                pca.fit(trainset)
                return_data['pca'][predict_region] = pca

                for latent_dim in pca_latent_dim_list:
                    return_data['X'][predict_region][f'pca_{latent_dim}'] = {}
                    train_pca = np.dot(check_array(trainset) - pca.mean_, pca.components_[:latent_dim].T)
                    test_pca = np.dot(check_array(testset) - pca.mean_, pca.components_[:latent_dim].T)
                    train_pca = pd.DataFrame(train_pca)
                    test_pca = pd.DataFrame(test_pca)

                    train_X = train_pca.set_index(trainset.index)
                    test_X = test_pca.set_index(testset.index)

                    return_data['X'][predict_region][f'pca_{latent_dim}']['train'] = train_X
                    return_data['X'][predict_region][f'pca_{latent_dim}']['test'] = test_X

        elif df_type == 'numeric':
            regions = list(self.wrf['train'].keys())
            ########### region tuning 시 바꿔야 할 부분 #############

            for i, region in enumerate(regions):
                if region in self.rm_regions:
                    print('numeric - remove region: ', region)
                else:
                    train_flat.append(self.wrf['train'][region])
                    test_flat.append(self.wrf['test'][region])

            #####################################################

            trainset = pd.concat(train_flat, axis=1)
            testset = pd.concat(test_flat, axis=1)

            pca = PCA(svd_solver='full', random_state=self.seed)
            pca.fit(trainset)
            return_data['pca'] = pca

            for latent_dim in pca_latent_dim_list:
                return_data['X'][f'pca_{latent_dim}'] = {}
                train_pca = np.dot(check_array(trainset) - pca.mean_, pca.components_[:latent_dim].T)
                test_pca = np.dot(check_array(testset) - pca.mean_, pca.components_[:latent_dim].T)
                train_pca = pd.DataFrame(train_pca)
                test_pca = pd.DataFrame(test_pca)

                train_X = train_pca.set_index(trainset.index)
                test_X = test_pca.set_index(testset.index)

                return_data['X'][f'pca_{latent_dim}']['train'] = train_X
                return_data['X'][f'pca_{latent_dim}']['test'] = test_X

        else:
            if df_type == 'fnl':
                train_flat.append(self.ewkr['train']['chinese_region'])
                test_flat.append(self.ewkr['test']['chinese_region'])

            trainset = pd.concat(train_flat, axis=1)
            testset = pd.concat(test_flat, axis=1)

            pca = PCA(svd_solver='full', random_state=self.seed)
            pca.fit(trainset)
            return_data['pca'] = pca

            for latent_dim in pca_latent_dim_list:
                return_data['X'][f'pca_{latent_dim}'] = {}
                train_pca = np.dot(check_array(trainset) - pca.mean_, pca.components_[:latent_dim].T)
                test_pca = np.dot(check_array(testset) - pca.mean_, pca.components_[:latent_dim].T)
                train_pca = pd.DataFrame(train_pca)
                test_pca = pd.DataFrame(test_pca)

                train_X = train_pca.set_index(trainset.index)
                test_X = test_pca.set_index(testset.index)

                return_data['X'][f'pca_{latent_dim}']['train'] = train_X
                return_data['X'][f'pca_{latent_dim}']['test'] = test_X

        return return_data
