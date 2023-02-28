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


class RMMakeNIERDataset(ABC):
    def __init__(self, reset_db=False, start_date=20170301, until_date=20220228, test_date=20210301, remove_region=None, seed=999,
                 preprocess_root='../dataset/d5', save_processed_data=True, run_pca=False):
        super(RMMakeNIERDataset, self).__init__()
        
        self.reset_db = reset_db
        self.preprocess_root = preprocess_root
        self.start_date = start_date
        self.until_date = until_date
        self.save_processed_data = save_processed_data
        self.remove_region = remove_region
        self.seed = seed
        self.test_date = test_date
        self.preprocess()

        if run_pca:
            self.handle_pca()
        else:
            if remove_region is None:
                save_path = 'all_region'
            else:
                save_path = '-'.join(remove_region)
            self.final_data = load_data(os.path.join(self.preprocess_root, f"{save_path}.pkl"))
            # 지역_[].pkl
            # R4_77_R4_69.pkl
            # all_region.pkl

    def preprocess(self):
        if self.save_processed_data:
            obs_df, fnl_df, wrf_df, cmaq_df, cwdb_df, ewkr_df = self._select_by_date()

            obs_train, obs_test, obs_scaler = self._preprocess_df(obs_df, 'obs', test_date=self.test_date)
            cw_train, cw_test, cw_scaler = self._preprocess_df(cwdb_df, 'cwdb', test_date=self.test_date)
            ewkr_train, ewkr_test, ewkr_scaler = self._preprocess_df(ewkr_df, 'ewkr', test_date=self.test_date)
            fnl_train, fnl_test, fnl_scaler = self._preprocess_df(fnl_df, 'fnl', test_date=self.test_date)
            wrf_train, wrf_test, wrf_scaler = self._preprocess_df(wrf_df, 'numeric', test_date=self.test_date)
            cmaq_train, cmaq_test, cmaq_scaler = self._preprocess_df(cmaq_df, 'numeric', test_date=self.test_date)

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
                scaler=cmaq_scaler
            )

            # self._save_preprocessed_v1(obs, cw, ewkr, fnl, wrf, cmaq)
        else:
            obs, cw, ewkr, fnl, wrf, cmaq = self._load_preprocessed_v1()
            # pass

        self.obs = obs
        self.cw = cw
        self.ewkr = ewkr
        self.fnl = fnl
        self.wrf = wrf
        self.cmaq = cmaq

    def _load_preprocessed_v1(self):
        start_year = str(self.start_date)[2:4]
        test_year = str(self.until_date)[2:4]

        obs = load_data(os.path.join(self.preprocess_root, f'um_obs_R4_{start_year}_to_{test_year}_221018.pkl'))
        cw = load_data(os.path.join(self.preprocess_root, f'um_cw_R4_{start_year}_to_{test_year}_221018.pkl'))
        ewkr = load_data(os.path.join(self.preprocess_root, f'um_ewkr_R4_{start_year}_to_{test_year}_221018.pkl'))
        fnl = load_data(os.path.join(self.preprocess_root, f'um_fnl_R4_{start_year}_to_{test_year}_221018.pkl'))
        wrf = load_data(os.path.join(self.preprocess_root, f'um_wrf_R4_{start_year}_to_{test_year}_221018.pkl'))
        cmaq = load_data(os.path.join(self.preprocess_root, f'um_cmaq_R4_{start_year}_to_{test_year}_221018.pkl'))

        return obs, cw, ewkr, fnl, wrf, cmaq

    def _save_preprocessed_v1(self, obs, cw, ewkr, fnl, wrf, cmaq):
        start_year = str(self.start_date)[2:4]
        test_year = str(self.until_date)[2:4]

        save_data(obs, self.preprocess_root, filename=f'um_obs_R4_{start_year}_to_{test_year}_221018.pkl')
        save_data(cw, self.preprocess_root, filename=f'um_cw_R4_{start_year}_to_{test_year}_221018.pkl')
        save_data(ewkr, self.preprocess_root, filename=f'um_ewkr_R4_{start_year}_to_{test_year}_221018.pkl')
        save_data(fnl, self.preprocess_root, filename=f'um_fnl_R4_{start_year}_to_{test_year}_221018.pkl')
        save_data(wrf, self.preprocess_root, filename=f'um_wrf_R4_{start_year}_to_{test_year}_221018.pkl')
        save_data(cmaq, self.preprocess_root, filename=f'um_cmaq_R4_{start_year}_to_{test_year}_221018.pkl')

    def _check_data(self, obs_df, fnl_df, wrf_df, cmaq_df):
        s_date = str(self.start_date)[0:4] + '-' + str(self.start_date)[4:6] + '-' + str(self.start_date)[6:8]
        u_date = str(self.until_date)[0:4] + '-' + str(self.until_date)[4:6] + '-' + str(self.until_date)[6:8]
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
                                                                      root_dir=self.preprocess_root)

        ############## 날짜 튜닝 시 작업해야할 부분 ######################

        obs_df = obs_df[(obs_df['RAW_DATE'] >= self.start_date) & (obs_df['RAW_DATE'] <= self.until_date)]
        fnl_df = fnl_df[(fnl_df['RAW_DATE'] >= self.start_date) & (fnl_df['RAW_DATE'] <= self.until_date)]
        wrf_df = wrf_df[(wrf_df['RAW_DATE'] >= self.start_date) & (wrf_df['RAW_DATE'] <= self.until_date)]
        cmaq_df = cmaq_df[(cmaq_df['RAW_DATE'] >= self.start_date) & (cmaq_df['RAW_DATE'] <= self.until_date)]
        cwdb_df = cwdb_df[(cwdb_df['RAW_DATE'] >= self.start_date) & (cwdb_df['RAW_DATE'] <= self.until_date)]
        ewkr_df = ewkr_df[(ewkr_df['RAW_DATE'] >= self.start_date) & (ewkr_df['RAW_DATE'] <= self.until_date)]
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

    def _preprocess_df(self, df, df_type, test_date=20210301):
        assert df_type in ['obs', 'fnl', 'numeric', 'cwdb', 'ewkr'], 'data_type: invalid parameter %s' % df_type
        if df_type == 'cwdb':
            df = region_flatten_cwdb(df, df_type)
            df = df.reset_index()

        train_df = df[df.RAW_DATE < test_date]
        test_df = df[df.RAW_DATE >= test_date]
        if df_type == 'numeric':
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

        # Concatenate chinese regions (< R4_59)
        if len(train_flat_df) > 0:
            chinese_region_train = pd.concat(train_flat_df, axis=1)
            chinese_region_test = pd.concat(test_flat_df, axis=1)

            train_region_dict['chinese_region'] = chinese_region_train
            test_region_dict['chinese_region'] = chinese_region_test

        return train_region_dict, test_region_dict, scaler

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
        if self.remove_region is None:
            save_path = 'all_region'
        else:
            save_path = '-'.join(self.remove_region)
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
            if region in self.remove_region:
                print('remove region: ', region)
            else:
                train_flat.append(main_df['train'][region])
                test_flat.append(main_df['test'][region])

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
                if region in self.remove_region:
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
