import pandas as pd
import numpy as np

import pymysql
from sqlalchemy import create_engine
# from .utils import read_yaml
pymysql.install_as_MySQLdb()

import os

pd.set_option('display.max_rows', 100)


inverse_transform = lambda scaled, var, mean : scaled * var + mean

def db_to_pkl(get_data=False, root_dir='./db_save'):
    # yaml_file = read_yaml("../../static.yaml")
    # db_master = yaml_file['DBMaster']
    db_master = "mysql+mysqldb://root:" + "root12" + "@166.104.185.217:32770/AIMTFS_R4"
    if get_data:
        engine = create_engine(db_master,
                               encoding='utf-8')
        conn = engine.connect()

        wrf_df = pd.read_sql("SELECT `WRF_UM1`.* FROM `WRF_UM1`", engine)
        cmaq_df = pd.read_sql("SELECT `CMAQ_UM1`.* FROM `CMAQ_UM1`", engine)

        # drop not use columns
        wrf_df = wrf_df.drop(['INSERT_DATE', 'UPDATE_DATE', 'WRF_SR'], axis=1)
        cmaq_df = cmaq_df.drop(['INSERT_DATE', 'UPDATE_DATE'], axis=1)

        obs_df = pd.read_sql(
            "SELECT `OBS_UM1`.*, `DATE_INFO`.`WEEK_NO_KR`,`DATE_INFO`.`WEEK_NO_CN` FROM `OBS_UM1` JOIN `DATE_INFO` ON `OBS_UM1`.`RAW_DATE`=`DATE_INFO`.`RAW_DATE`",
            engine)
        fnl_df = pd.read_sql("SELECT `FNL`.* FROM `FNL`", engine)
        cwdb_df = pd.read_sql("SELECT `CWDB_UM1`.* FROM `CWDB_UM1`", engine)
        ewkr_df = pd.read_sql("SELECT `EWGI_KRBI`.* FROM `EWGI_KRBI`", engine)

        # drop not use columns
        obs_df = obs_df.drop(['INSERT_DATE', 'UPDATE_DATE', 'ASOS_VS', 'ASOS_SI_HR1'], axis=1)
        fnl_df = fnl_df.drop(['INSERT_DATE', 'UPDATE_DATE'], axis=1)
        cwdb_df = cwdb_df.drop(['INSERT_DATE', 'UPDATE_DATE'], axis=1)
        ewkr_df = ewkr_df.drop(['INSERT_DATE', 'UPDATE_DATE'], axis=1)
        wrf_df = wrf_df.drop('FORECAST_DATE', axis=1)
        cmaq_df = cmaq_df.drop('FORECAST_DATE', axis=1)

        # drop nodata region --> R4_53, R4_54, R4_58
        obs_df = obs_df[(obs_df['REGION_CODE'] != 'R4_53') & (obs_df['REGION_CODE'] != 'R4_54') & (
                obs_df['REGION_CODE'] != 'R4_58')]
        fnl_df = fnl_df[(fnl_df['REGION_CODE'] != 'R4_53') & (fnl_df['REGION_CODE'] != 'R4_54') & (
                fnl_df['REGION_CODE'] != 'R4_58')]
        cwdb_df = cwdb_df[(cwdb_df['SOURCE_REGION_CODE'] != 'R4_53') & (cwdb_df['SOURCE_REGION_CODE'] != 'R4_54') & (
                cwdb_df['SOURCE_REGION_CODE'] != 'R4_58')]

        # save df to pkl
        cwdb_df.to_pickle(os.path.join(root_dir, "cwdb_r4_um.pkl"))
        ewkr_df.to_pickle(os.path.join(root_dir, "ewkr_r4.pkl"))
        obs_df.to_pickle(os.path.join(root_dir, "obs_r4_um.pkl"))
        fnl_df.to_pickle(os.path.join(root_dir, "fnl_r4.pkl"))
        wrf_df.to_pickle(os.path.join(root_dir, "wrf_r4_um.pkl"))
        cmaq_df.to_pickle(os.path.join(root_dir, "cmaq_r4_um.pkl"))

    else:
        obs_df = pd.read_pickle(os.path.join(root_dir, "obs_r4_um.pkl"))
        fnl_df = pd.read_pickle(os.path.join(root_dir, "fnl_r4.pkl"))
        cwdb_df = pd.read_pickle(os.path.join(root_dir, "cwdb_r4_um.pkl"))
        ewkr_df = pd.read_pickle(os.path.join(root_dir, "ewkr_r4.pkl"))
        wrf_df = pd.read_pickle(os.path.join(root_dir, "wrf_r4_um.pkl"))
        cmaq_df = pd.read_pickle(os.path.join(root_dir, "cmaq_r4_um.pkl"))

    return obs_df, fnl_df, wrf_df, cmaq_df, cwdb_df, ewkr_df


def oversampling(data, label, seed=1):
    np.random.seed(seed)
    idx = np.where(label == 1)[0]  # minor class is 1
    sampled_idx = np.random.choice(idx, label.shape[0] - (2 * idx.shape[0]), replace=True)
    sampled_data = data[sampled_idx]
    sampled_label = label[sampled_idx]

    n_data = data.shape[0] + sampled_data.shape[0]
    n_feature = data.shape[1]

    new_data = np.zeros(n_data * n_feature).reshape(n_data, n_feature)
    new_label = np.zeros(n_data)

    new_data[:data.shape[0]] = data
    new_data[data.shape[0]:] = sampled_data
    new_label[:data.shape[0]] = label
    new_label[data.shape[0]:] = sampled_label

    return new_data, new_label


def window_shifting(df, window_size, types, threshold, test_date, filetype, region_names, seed=1):
    X = []
    X_m = []

    y_3 = []
    y_4 = []
    y_5 = []
    y_6 = []
    #     y_7 = []

    i = 3
    test_idx = 0
    idx = 0
    flag = True
    while (True):
        # Out of bound 예외처리
        if i - window_size < 0:
            i += 4
            continue

        if i + 28 + 4 >= len(df):
            break

        # +2 는 15시 예측
        tx = df.iloc[i - window_size:i + 4].isna().sum().sum()
        if tx > 0:
            print('missing!')
        X.append(np.array(df.iloc[i - window_size:i + 4]))
        #         X_m.append(np.array(month_df.iloc[i-window_size:i+4]))

        if flag and str(df.iloc[i - window_size:i + 4].index[0][0]) == test_date:
            flag = False
            test_idx = idx

        y_3.append(np.array(df.iloc[i + 12 + 3:i + 12 + 3 + 1]))
        y_4.append(np.array(df.iloc[i + 16 + 3:i + 16 + 3 + 1]))
        y_5.append(np.array(df.iloc[i + 20 + 3:i + 20 + 3 + 1]))
        y_6.append(np.array(df.iloc[i + 24 + 3:i + 24 + 3 + 1]))

        i += 4
        idx += 1

    print('test_idx = %d' % test_idx)
    X = np.array(X)
    X_m = np.array(X_m)

    np.random.seed(seed)

    train_indice = np.arange(test_idx)
    test_indice = np.arange(test_idx, X.shape[0])

    X_train, X_test = X[train_indice], X[test_indice]
    y_dict = {region: {'cls': {'train': [], 'test': []}, 'reg': {'train': [], 'test': []}} for region in region_names}

    for region in region_names:
        for y in [y_3, y_4, y_5, y_6]:
            targets = np.array(y)[:, :, df.columns.get_loc('[%s]%s' % (region, types))].reshape(-1)
            y_dict[region]['cls']['train'].append(
                np.array([1 if target > threshold else 0 for target in targets[train_indice]]))
            y_dict[region]['reg']['train'].append(targets[train_indice])
            y_dict[region]['cls']['test'].append(
                np.array([1 if target > threshold else 0 for target in targets[test_indice]]))
            y_dict[region]['reg']['test'].append(targets[test_indice])

    return X_train, X_test, y_dict


def region_flatten(df, df_type):
    flat_df = []
    if df_type == 'obs':
        common_cols = ['RAW_DATE', 'RAW_TIME', 'WEEK_NO_KR', 'WEEK_NO_CN', 'RAW_MONTH']
    if df_type == 'fnl':
        common_cols = ['RAW_DATE', 'RAW_TIME']
    if df_type == 'numeric':
        common_cols = ['RAW_DATE', 'RAW_FDAY', 'RAW_TIME']
    region = df['REGION_CODE'].unique().tolist()
    for i, reg in enumerate(region):
        region_df = df.loc[df['REGION_CODE'] == reg]
        rename_idx = len(region_df.columns) - len(common_cols)
        region_df = region_df.drop(['REGION_CODE'], axis=1)
        if i > 0:
            region_df = region_df.drop(common_cols, axis=1)
        for col in region_df.columns[-(rename_idx - 1):]:
            region_df = region_df.rename(columns={col: '[' + reg + ']' + col}, errors="raise")
        region_df = region_df.reset_index(drop=True)
        flat_df.append(region_df)
    flatten_df = pd.concat(flat_df, axis=1)
    if df_type == 'obs':
        common_cols = ['RAW_DATE', 'RAW_TIME']
    flatten_df = flatten_df.set_index(common_cols)
    return flatten_df


def region_flatten_cwdb(df, df_type):
    flat_df = []
    common_cols = ['REGION_CODE', 'RAW_DATE', 'RAW_TIME']
    region = df['REGION_CODE'].unique()
    source_region = df['SOURCE_REGION_CODE'].unique()
    for reg in region:
        reg_flat_df = []
        region_df = df.loc[df['REGION_CODE'] == reg]
        for i, s_reg in enumerate(source_region):
            s_region_df = region_df.loc[region_df['SOURCE_REGION_CODE'] == s_reg].drop(['SOURCE_REGION_CODE'], axis=1)
            rename_idx = len(s_region_df.columns) - len(common_cols)
            if i > 0:
                s_region_df = s_region_df.drop(common_cols, axis=1)
            for col in s_region_df.columns[-rename_idx:]:
                s_region_df = s_region_df.rename(columns={col: '[' + s_reg + ']' + col}, errors="raise")
            s_region_df = s_region_df.reset_index(drop=True)
            reg_flat_df.append(s_region_df)
        reg_df = pd.concat(reg_flat_df, axis=1)
        flat_df.append(reg_df)
    flatten_df = pd.concat(flat_df, axis=0)
    flatten_df = flatten_df.reset_index(drop=True)
    flatten_df = flatten_df.set_index(common_cols)
    return flatten_df


def preprocessing(df, data_type):
    assert data_type in ['obs', 'fnl', 'numeric', 'cwdb', 'ewkr'], 'data_type: invalid parameter %s' % data_type
    if data_type == 'obs':
        df = region_flatten(df, data_type)
        df = df.reset_index()
        df = df.set_index(['RAW_DATE', 'RAW_TIME'])
        month_df = df.pop('RAW_MONTH')
        unique, inverse = np.unique(month_df, return_inverse=True)
        month_df = np.eye(unique.shape[0])[inverse]
        month_df = pd.DataFrame(month_df)
        return df, month_df
    if data_type == 'fnl':
        df = region_flatten(df, data_type)
    if data_type == 'numeric':
        df = region_flatten(df, data_type)
        df = df.reset_index()
        df = df.set_index(['RAW_DATE', 'RAW_TIME'])
    if data_type == 'cwdb':
        df = region_flatten_cwdb(df, data_type)
    if data_type == 'ewkr':
        df = df.drop('REGION_CODE', axis=1)
        df = df.set_index(['RAW_DATE', 'RAW_TIME'])
    return df


def one_hot(array, flag=False):
    unique, inverse = np.unique(array, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    if flag:
        onehot = pd.DataFrame(onehot)
    return onehot


def df_concat(df1, df2):
    df = pd.concat([df1, df2], axis=1)
    return df


def pm_labeling(obs_df, other_df, data_type):
    # extract OBS R08(seoul) PM10,PM2.5 for label
    pm10_col = ['[R4_' + str(region) + ']PM10' for region in range(59, 78)]
    pm25_col = ['[R4_' + str(region) + ']PM25' for region in range(59, 78)]
    pm10 = obs_df[pm10_col].reset_index()
    pm25 = obs_df[pm25_col].reset_index()
    for time in ['03', '09', '21']:
        pm10.loc[pm10['RAW_TIME'] == time, pm10_col] = np.NaN
        pm25.loc[pm25['RAW_TIME'] == time, pm25_col] = np.NaN
    pm25 = pm25.drop(['RAW_DATE', 'RAW_TIME'], axis=1)
    label_df = pd.concat([pm10, pm25], axis=1)
    label_df = label_df.set_index(['RAW_DATE', 'RAW_TIME'])

    # fnl case
    if data_type == 'fnl':
        fnl_df = pd.concat([other_df, label_df], axis=1)
        return fnl_df

    # wrf+cmaq case
    elif data_type == 'numeric':
        date_list = list(other_df.index.unique(level='RAW_DATE'))
        label_df = label_df.reset_index()
        label_list = []
        for date in date_list:
            label_origin_df = label_df[label_df['RAW_DATE'] == int(date)]
            for i in range(7):
                label_list.append(label_origin_df)
        label_numeric_df = pd.concat(label_list)
        label_numeric_df = label_numeric_df.set_index(['RAW_DATE', 'RAW_TIME'])
        other_df = other_df.reset_index()
        other_df = other_df.set_index(['RAW_DATE', 'RAW_TIME'])
        numeric_df = pd.concat([other_df, label_numeric_df], axis=1)
        return numeric_df

    else:
        return 0


def get_nan_percent(df, feature):
    percent = 100 * (df[feature].isna().sum() / df[feature].shape[0])
    return percent


def get_nandate(feature_name, df):
    percent = get_nan_percent(df, feature_name)
    nan_df = df.loc[(df[feature_name].isna())]
    nan_date = nan_df.loc[:, ['RAW_DATE', 'RAW_TIME', 'RAW_FDAY']]
    return nan_date, percent


def repeat_2020(df, times):
    start_d = df.index.get_loc(20200101).start
    stop_d = df.index.get_loc(20210228).stop
    df_2020 = df.iloc[start_d:stop_d]
    df_2020 = pd.concat([df_2020] * times)
    df_past = df.iloc[:start_d]
    df_test = df.iloc[stop_d:]
    result_df = pd.concat([df_past, df_2020, df_test])
    return result_df


def repeat_2020_cwdb(df, times):
    all_region = []
    for region in range(59, 78):
        df_region = df.loc['R4_' + str(region)]
        start_d = df_region.index.get_loc(20200101).start
        stop_d = df_region.index.get_loc(20210228).stop
        df_2020 = df_region.iloc[start_d:stop_d]
        df_2020 = pd.concat([df_2020] * times)
        df_past = df_region.iloc[:start_d]
        df_test = df_region.iloc[stop_d:]
        result_df = pd.concat([df_past, df_2020, df_test])

        reg_list = ['R4_' + str(region)] * result_df.shape[0]
        reg_list = np.array(reg_list)
        result_df = result_df.reset_index()
        result_df.insert(0, 'REGION_CODE', reg_list)
        result_df = result_df.set_index(['REGION_CODE', 'RAW_DATE', 'RAW_TIME'])
        all_region.append(result_df)
    cwdb_repeat_df = pd.concat(all_region, axis=0)
    return cwdb_repeat_df
