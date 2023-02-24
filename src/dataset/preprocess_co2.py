#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2023/02/09 2:04 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : preprocess_co2.py
# @Software  : PyCharm

import pandas as pd
import numpy as np
import sys

from ..utils import load_data, save_data

from sklearn.preprocessing import StandardScaler


def _co2_read(start_date, end_date, root_dir='/workspace/local/R5/sia_co2'):
    anmyendo_df = pd.read_parquet(f'{root_dir}/final_df_anmyeondo.parquet.snappy')
    gosan_df = pd.read_parquet(f'{root_dir}/final_df_gosan.parquet.snappy')
    anmyendo_df['region'] = 'R4_65'
    gosan_df['region'] = 'R4_77'

    raw_df = pd.concat([anmyendo_df, gosan_df], axis=0)

    hour = raw_df['OBS_DATE'].dt.year.astype(str)
    month = raw_df['OBS_DATE'].dt.month.astype(str).str.zfill(2)
    day = raw_df['OBS_DATE'].dt.day.astype(str).str.zfill(2)
    raw_date = hour + month + day
    raw_df['RAW_DATE'] = raw_date.astype(int)
    raw_df['RAW_TIME'] = raw_df['OBS_DATE'].dt.hour.astype(str).str.zfill(2)
    raw_df = raw_df[(raw_df['RAW_DATE'] >= start_date) & (raw_df['RAW_DATE'] <= end_date)]
    raw_df.column = raw_df.rename(columns={'clean_inter1_6h_CO2': f'clean_inter1_6h_CO2',
                                           'fin_rolling_12': f'fin_rolling_12',
                                           'fin_rolling_20': f'fin_rolling_20'}, inplace=True)
    raw_df = raw_df[['RAW_DATE', 'RAW_TIME', f'clean_inter1_6h_CO2', f'fin_rolling_12', f'fin_rolling_20', 'region']]
    #     raw_df = raw_df.set_index(['RAW_DATE','RAW_TIME'])
    #     raw_df['region'] = region
    return raw_df


def _make_co2_dataset(start_date=20190101, end_date=20211231, test_date=20210101, root_dir='/workspace/local/R5/sia_co2'):
    co2_df = _co2_read(start_date, end_date, root_dir)

    train_df = co2_df[co2_df.RAW_DATE < test_date]
    test_df = co2_df[co2_df.RAW_DATE >= test_date]

    index_cols = ['RAW_DATE', 'RAW_TIME']
    regions = train_df.region.unique().tolist()
    train_df_region = train_df.region
    test_df_region = test_df.region

    train_df = train_df.set_index(index_cols)
    test_df = test_df.set_index(index_cols)
    # train_df = train_df.drop(['region'], axis=1)
    # test_df = test_df.drop(['region'], axis=1)
    scale_indices = ['clean_inter1_6h_CO2', 'fin_rolling_12', 'fin_rolling_20']
    scaler = StandardScaler()
    train_df_scaled = scaler.fit_transform(train_df[scale_indices])
    test_df_scaled = scaler.transform(test_df[scale_indices])

    train_df[scale_indices] = train_df_scaled
    test_df[scale_indices] = test_df_scaled

    regions = train_df.region.unique().tolist()

    train_region_dict = {}
    test_region_dict = {}

    for i, region in enumerate(regions):
        region_id = int(region.split("_")[1])
        train_d = train_df.loc[train_df.region == region]
        test_d = test_df.loc[test_df.region == region]
        train_d = train_d.drop(['region'], axis=1)
        test_d = test_d.drop(['region'], axis=1)

        train_region_dict[region] = train_d
        test_region_dict[region] = test_d

    return train_region_dict, test_region_dict, scaler


def _save_co2_dataset(train, test, scaler, preprocess_root, start_year, test_year):
    obj = dict(
        train=train,
        test=test,
        scaler=scaler
    )

    save_data(obj, preprocess_root, filename=f'co2_{start_year}_to_{test_year}.pkl')


def create_co2_data(start_date=20190101, end_date=20211231, test_date=20210101, preprocess_root='../dataset/d5', data_root='/workspace/local/R5/sia_co2'):
    start_year, test_year = str(start_date)[2:4], str(end_date)[2:4]
    train_dict, test_dict, scaler = _make_co2_dataset(start_date, end_date, test_date, root_dir=data_root)
    _save_co2_dataset(train_dict, test_dict, scaler, preprocess_root, start_year, test_year)


