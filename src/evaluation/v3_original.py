#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2023/03/29 10:47 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : v3_original.py
# @Software  : PyCharm

import numpy as np
import pandas as pd

import os
from datetime import date, timedelta


from utils import load_data, save_data, obs_window_shifting, window_shifting_V3, get_relative_date, pod_score, accuracy_score, far_score, unpickling
from models import DoubleInceptionCRNN, DoubleInceptionModel, SingleInceptionCRNN, SingleInceptionModel
from model_loader import best_cnnrnn_loader, best_cnn_loader
from data import get_v3_data

from sklearn.metrics import f1_score

import time
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm_notebook as tqdm

DEVICE = "cuda:1"
HORIZONS = range(1, 5)
pm = 'pm10'
SAVE_FILE = 'pm10_re.pkl'

DAYS=14

df = pd.read_csv('/workspace/all_data/dust/HM_2021_R3/processed_data/obs_R3_0714.csv')
df.set_index(['[Common]RAW_DATE', '[Common]RAW_TIME'], inplace=True)


def do_test(model, test_loader, dtype, is_reg, device):
    start = time.time()

    model.to(device)
    model.eval()
    scores = torch.zeros(0)
    labels = []
    #     print('model load', time.time() - start)
    #     start = time.time()

    if dtype == 'numeric':

        (obs_x, obs_y), (fnl_x, fnl_y), (num_x, num_y) = test_loader
        obs_x, obs_y = torch.tensor(obs_x).to(device).float(), torch.tensor(obs_y).to(device).long()
        fnl_x, fnl_y = torch.tensor(fnl_x).to(device).float(), torch.tensor(fnl_y).to(device).long()
        num_x, num_y = torch.tensor(num_x).to(device).float(), torch.tensor(num_y).to(device).long()

        #         print('num data to gpu', time.time() - start)
        #         start = time.time()

        #         print(obs_x.shape, fnl_x.shape, num_x.shape)

        output = model(obs_x, fnl_x, num_x)

        #         print('model inference', time.time() - start)
        #         start = time.time()

        if is_reg:
            output = output[:, 0]
            score = output
        else:
            score = F.softmax(output, dim=1)

        scores = score.cpu().detach()
        labels.append(obs_y.cpu().detach())

    #         print('score fetch', time.time() - start)
    #         start = time.time()
    else:
        (obs_x, obs_y), (fnl_x, fnl_y) = test_loader
        obs_x, obs_y = torch.tensor(obs_x).to(device).float(), torch.tensor(obs_y).to(device).long()
        fnl_x, fnl_y = torch.tensor(fnl_x).to(device).float(), torch.tensor(fnl_y).to(device).long()

        #         print('data to gpu', time.time() - start)
        #         start = time.time()

        #         print(obs_x.shape, fnl_x.shape)

        output = model(obs_x, fnl_x, None)

        #         print('model inference', time.time() - start)
        #         start = time.time()
        if is_reg:
            score = output[:, 0]
        else:
            score = F.softmax(output, dim=1)

        scores = score.cpu().detach()
        labels.append(obs_y.cpu().detach())

    #         print('score fetch', time.time() - start)
    #         start = time.time()
    labels = torch.tensor(labels)
    scores = scores.numpy()
    labels = labels.numpy()

    return scores, labels


def do_ensemble(test_result_dict):
    pass


def ensemble_test(models, results, date_value, device):
    start = time.time()
    output = None

    entire_result_dict = dict()

    if type(models) == list:

        for model, result in zip(models, results):
            param = result["param"]
            threshold = result["threshold"]
            dtype = param["dtype"]
            is_reg = param["reg"]
            season = param["season"]
            if 'model_name' in param.keys():
                model_name = param["model_name"]
            else:
                model_name = param["model_type"]

            if season == 'year':
                season_name = season
            else:
                season_name = 'seasonal'

            model_type = "reg" if param["reg"] else "clf"
            model_repr = f"{model_name}_{model_type}_{season_name}"

            #             print(param)

            try:
                test_data = get_v3_data(param, date_value, df=df)
                test_score, test_label = do_test(model, test_data, dtype=dtype, is_reg=is_reg, device=device)

                pred = None
                label = None

                if is_reg:
                    score = None

                    if param['pm'] == 'PM10':
                        score = np.zeros(len(test_label))
                        score[test_score < 80.0] = 0
                        score[test_score > 80.0] = 1

                        label = np.zeros(len(test_label))
                        label[test_label < 80.0] = 0
                        label[test_label > 80.0] = 1
                    elif param['pm'] == 'PM25':
                        score = np.zeros(len(test_label))
                        score[test_score < 35.0] = 0
                        score[test_score > 35.0] = 1

                        label = np.zeros(len(test_label))
                        label[test_label < 35.0] = 0
                        label[test_label > 35.0] = 1

                    entire_result_dict[model_repr] = dict(
                        pred=score[0],
                        pred2=test_score[0],
                        label=label[0],
                        is_reg=is_reg,
                        season=season,
                    )
                else:
                    pred = np.zeros(len(test_score))
                    label = test_label

                    pred[test_score[:, 1] < threshold] = 0
                    pred[test_score[:, 1] >= threshold] = 1

                    entire_result_dict[model_repr] = dict(
                        pred=pred[0],
                        pred2=test_score[:, 1][0],
                        label=label[0],
                        is_reg=is_reg,
                        season=season
                    )
            except AssertionError:
                pass


    else:
        param = results["param"]
        threshold = results["threshold"]
        dtype = param["dtype"]
        is_reg = param["reg"]
        season = param["season"]
        if 'model_name' in param.keys():
            model_name = param["model_name"]
        else:
            model_name = param["model_type"]

        if season == 'year':
            season_name = season
        else:
            season_name = 'seasonal'

        model_type = "reg" if param["reg"] else "clf"
        model_repr = f"{model_name}_{model_type}_{season_name}"

        try:
            test_score, test_label = do_test(models, get_v3_data(param, date_value, df=df), dtype=dtype, is_reg=is_reg,
                                             device=device)

            pred = None
            label = None

            if is_reg:
                score = None

                if param['pm'] == 'PM10':
                    score = np.zeros(len(test_label))
                    score[test_score < 80.0] = 0
                    score[test_score > 80.0] = 1

                    label = np.zeros(len(test_label))
                    label[test_label < 80.0] = 0
                    label[test_label > 80.0] = 1
                elif param['pm'] == 'PM25':
                    score = np.zeros(len(test_label))
                    score[test_score < 35.0] = 0
                    score[test_score > 35.0] = 1

                    label = np.zeros(len(test_label))
                    label[test_label < 35.0] = 0
                    label[test_label > 35.0] = 1

                entire_result_dict[model_repr] = dict(
                    pred=score[0],
                    pred2=test_score[0],
                    label=label[0],
                    is_reg=is_reg,
                    season=season
                )
            else:
                pred = np.zeros(len(test_score))
                label = test_label

                pred[test_score[:, 1] < threshold] = 0
                pred[test_score[:, 1] >= threshold] = 1

                entire_result_dict[model_repr] = dict(
                    pred=pred[0],
                    pred2=test_score[:, 1][0],
                    label=label[0],
                    is_reg=is_reg,
                    season=season
                )
        except AssertionError:
            pass

    return entire_result_dict


def run_v3_test(models, d, results):
    outputs = []
    isBreak = False

    for r in results:
        if d - timedelta(days=7) - timedelta(days=horizon) - timedelta(days=r['param']['lag']) < date(2019, 1,
                                                                                                      1) or d > date(
                2019, 12, 31):
            isBreak = True

    if isBreak:
        return -1, outputs
    else:
        for j in range(DAYS):  # validation - take best models from classification and regression

            dat = d - timedelta(days=j) - timedelta(days=horizon)

            for r in results:
                if dat - timedelta(days=r['param']['lag']) < date(2019, 1, 1) or d > date(2019, 12, 31):
                    isBreak = True

            if isBreak:
                break

            output = ensemble_test(models, results, date_value=_strinify_date(dat + timedelta(days=horizon)),
                                   device=DEVICE)
            if len(output.keys()) == 0:
                print('no outputs')
            else:
                outputs.append(output)

        # validation
        best_reg_idx, best_cls_idx, top3_reg_idx, top3_cls_idx, top3_idx = validation(
            outputs)  # top3 및 best 모델 index (models[index])

        # testing
        #         print(best_reg_idx, best_cls_idx, top3_reg_idx, top3_cls_idx, top3_idx)
        best_reg_output = ensemble_test(models[best_reg_idx], results[best_reg_idx], date_value=_strinify_date(d),
                                        device=DEVICE)
        best_cls_output = ensemble_test(models[best_cls_idx], results[best_cls_idx], date_value=_strinify_date(d),
                                        device=DEVICE)

        if len(best_reg_output.keys()) == 0:
            return -1, ()

        #         print([best_reg_output[dic] for dic in best_reg_output][0])

        best_reg_output = {'pred': [best_reg_output[dic] for dic in best_reg_output][0]['pred'],
                           'reg_result': [best_reg_output[dic] for dic in best_reg_output][0]['pred2'],
                           'label': [best_reg_output[dic] for dic in best_reg_output][0]['label'],
                           'is_reg': [best_reg_output[dic] for dic in best_reg_output][0]['is_reg']}
        best_cls_output = {'pred': [best_cls_output[dic] for dic in best_cls_output][0]['pred'],
                           'logit': [best_cls_output[dic] for dic in best_cls_output][0]['pred2'],
                           'label': [best_cls_output[dic] for dic in best_cls_output][0]['label'],
                           'is_reg': [best_cls_output[dic] for dic in best_cls_output][0]['is_reg']}

        # @TODO Do Ensemble
        ensemble_reg_output = ensemble_test([models[i] for i in top3_reg_idx], [results[i] for i in top3_reg_idx],
                                            date_value=_strinify_date(d), device=DEVICE)
        ensemble_cls_output = ensemble_test([models[i] for i in top3_cls_idx], [results[i] for i in top3_cls_idx],
                                            date_value=_strinify_date(d), device=DEVICE)
        ensemble_total = ensemble_test([models[i] for i in top3_idx], [results[i] for i in top3_idx],
                                       date_value=_strinify_date(d), device=DEVICE)

        if len(ensemble_reg_output.keys()) == 0:
            return -1, ()

        #         print([ensemble_reg_output[dic] for dic in ensemble_reg_output])

        ensemble_reg_return = [{'pred': [ensemble_reg_output[dic] for dic in ensemble_reg_output][i]['pred'],
                                'reg_result': [ensemble_reg_output[dic] for dic in ensemble_reg_output][i]['pred2'],
                                'label': [ensemble_reg_output[dic] for dic in ensemble_reg_output][i]['label'],
                                'is_reg': [ensemble_reg_output[dic] for dic in ensemble_reg_output][i]['is_reg']} for i
                               in range(3)]
        ensemble_cls_return = [{'pred': [ensemble_cls_output[dic] for dic in ensemble_cls_output][i]['pred'],
                                'logit': [ensemble_cls_output[dic] for dic in ensemble_cls_output][i]['pred2'],
                                'label': [ensemble_cls_output[dic] for dic in ensemble_cls_output][i]['label'],
                                'is_reg': [ensemble_cls_output[dic] for dic in ensemble_cls_output][i]['is_reg']} for i
                               in range(3)]
        ensemble_total_return = [{'pred': [ensemble_total[dic] for dic in ensemble_total][i]['pred'],
                                  'pred2': [ensemble_total[dic] for dic in ensemble_total][i]['pred2'],
                                  'label': [ensemble_total[dic] for dic in ensemble_total][i]['label'],
                                  'is_reg': [ensemble_total[dic] for dic in ensemble_total][i]['is_reg']} for i in
                                 range(3)]

        model_index = {
            'best_reg_idx': best_reg_idx,
            'best_cls_idx': best_cls_idx,
            'top3_reg_idx': top3_reg_idx,
            'top3_cls_idx': top3_cls_idx,
            'top3_idx': top3_idx
        }

        #         print(ensemble_reg_return)

        return j, (
        best_reg_output, best_cls_output, ensemble_reg_return, ensemble_cls_return, ensemble_total_return, model_index)


def validation(outputs):
    reg_models = ['rnn_reg_year', 'cnn_reg_year', 'rnn_reg_seasonal', 'cnn_reg_seasonal']  # 2i + 1
    cls_models = ['rnn_clf_year', 'cnn_clf_year', 'rnn_clf_seasonal', 'cnn_clf_seasonal']  # 2i - 1
    total_models = ['rnn_clf_year', 'rnn_reg_year', 'cnn_clf_year', 'cnn_reg_year', 'rnn_clf_seasonal',
                    'rnn_reg_seasonal', 'cnn_clf_seasonal', 'cnn_reg_seasonal']

    reg_validation = {}
    cls_validation = {}
    total_validation = {}

    for modeln in total_models:
        if modeln in reg_models:
            reg_validation[modeln] = {'label': [], 'pred': []}
        else:
            cls_validation[modeln] = {'label': [], 'pred': []}
        total_validation[modeln] = {'label': [], 'pred': []}

    for o in outputs:
        for modeln in total_models:
            if modeln in reg_models:
                reg_validation[modeln]['label'].append(o[modeln]['label'])
                reg_validation[modeln]['pred'].append(o[modeln]['pred'])
            else:
                cls_validation[modeln]['label'].append(o[modeln]['label'])
                cls_validation[modeln]['pred'].append(o[modeln]['pred'])

            total_validation[modeln]['label'].append(o[modeln]['label'])
            total_validation[modeln]['pred'].append(o[modeln]['pred'])

    best_reg_idx = 0
    best_clf_idx = 0
    top3_reg_idx = None
    top3_clf_idx = None

    best_f1 = -1
    f1_list = []
    for i, v in enumerate(reg_validation):
        test = reg_validation[v]
        f1 = f1_score(test['label'], test['pred'])
        test['f1'] = f1
        f1_list.append(f1)
        if best_f1 < f1:
            best_f1 = f1
            best_reg_idx = i * 2 + 1
    f1_list = np.array(f1_list)

    args = f1_list.argsort()
    top3_reg_idx = args[-3:] * 2 + 1

    best_f1 = -1
    f1_list = []
    for i, v in enumerate(cls_validation):
        test = cls_validation[v]
        f1 = f1_score(test['label'], test['pred'])
        test['f1'] = f1
        f1_list.append(f1)
        if best_f1 < f1:
            best_f1 = f1
            best_clf_idx = i * 2
    f1_list = np.array(f1_list)

    args = f1_list.argsort()
    top3_clf_idx = args[-3:] * 2

    best_f1 = -1
    f1_list = []
    for i, v in enumerate(total_validation):
        test = total_validation[v]
        f1 = f1_score(test['label'], test['pred'])
        f1_list.append(f1)
    f1_list = np.array(f1_list)

    args = f1_list.argsort()
    top3_idx = args[-3:]

    return best_reg_idx, best_clf_idx, top3_reg_idx, top3_clf_idx, top3_idx


def _strinify_date(d):
    return d.isoformat().replace('-', '')


if __name__ == "__main__":
    # example: 특정 날짜에서 예측 한다 했을 때 향후 1일~7일까지를 예측하는 모델 테스트
    # pm10 2.5 에 대해서 각 horizon 별 앙상블 모델 예측 결과 도출

    MODEL_PATH = '/workspace/all_data/dust/JHLEE11/save/'

    best_results = {}

    for horizon in tqdm(HORIZONS):  # 각 horizon 모델 마다 해당 날짜 데이터를 입력해서 inference
        best_results[horizon] = []

        cls_rnn, cls_rnn_res = best_cnnrnn_loader(horizon, 'year', 'kfold', pm, isReg=False, root_path=MODEL_PATH)
        reg_rnn, reg_rnn_res = best_cnnrnn_loader(horizon, 'year', 'kfold', pm, isReg=True, root_path=MODEL_PATH)
        cls_cnn, cls_cnn_res = best_cnn_loader(horizon, 'year', 'kfold', pm, isReg=False, root_path=MODEL_PATH)
        reg_cnn, reg_cnn_res = best_cnn_loader(horizon, 'year', 'kfold', pm, isReg=True, root_path=MODEL_PATH)
        #     print(cls_rnn_res['param'])

        reg_cnn_res['param']['model_name'] = 'cnn'
        reg_rnn_res['param']['model_name'] = 'rnn'

        day = date(2019, 1, 1)  # -> 예측 시점의 날짜라고 치자
        print('start horizon %d' % horizon)
        for _ in tqdm(range(365)):  # 2019년도 1년치 전부에 대해서 계산
            #         print(day)
            if day.month <= 3 and day.month >= 2:
                cls_rnn_s, cls_rnn_res_s = best_cnnrnn_loader(horizon, '0104', 'kfold', pm, isReg=False,
                                                              root_path=MODEL_PATH)
                reg_rnn_s, reg_rnn_res_s = best_cnnrnn_loader(horizon, '0104', 'kfold', pm, isReg=True,
                                                              root_path=MODEL_PATH)
                cls_cnn_s, cls_cnn_res_s = best_cnn_loader(horizon, '0104', 'kfold', pm, isReg=False,
                                                           root_path=MODEL_PATH)
                reg_cnn_s, reg_cnn_res_s = best_cnn_loader(horizon, '0104', 'kfold', pm, isReg=True,
                                                           root_path=MODEL_PATH)

            elif day.month <= 5 and day.month >= 4:
                cls_rnn_s, cls_rnn_res_s = best_cnnrnn_loader(horizon, '0306', 'kfold', pm, isReg=False,
                                                              root_path=MODEL_PATH)
                reg_rnn_s, reg_rnn_res_s = best_cnnrnn_loader(horizon, '0306', 'kfold', pm, isReg=True,
                                                              root_path=MODEL_PATH)
                cls_cnn_s, cls_cnn_res_s = best_cnn_loader(horizon, '0306', 'kfold', pm, isReg=False,
                                                           root_path=MODEL_PATH)
                reg_cnn_s, reg_cnn_res_s = best_cnn_loader(horizon, '0306', 'kfold', pm, isReg=True,
                                                           root_path=MODEL_PATH)

            elif day.month <= 7 and day.month >= 6:
                cls_rnn_s, cls_rnn_res_s = best_cnnrnn_loader(horizon, '0508', 'kfold', pm, isReg=False,
                                                              root_path=MODEL_PATH)
                reg_rnn_s, reg_rnn_res_s = best_cnnrnn_loader(horizon, '0508', 'kfold', pm, isReg=True,
                                                              root_path=MODEL_PATH)
                cls_cnn_s, cls_cnn_res_s = best_cnn_loader(horizon, '0508', 'kfold', pm, isReg=False,
                                                           root_path=MODEL_PATH)
                reg_cnn_s, reg_cnn_res_s = best_cnn_loader(horizon, '0508', 'kfold', pm, isReg=True,
                                                           root_path=MODEL_PATH)

            elif day.month <= 9 and day.month >= 8:
                cls_rnn_s, cls_rnn_res_s = best_cnnrnn_loader(horizon, '0710', 'kfold', pm, isReg=False,
                                                              root_path=MODEL_PATH)
                reg_rnn_s, reg_rnn_res_s = best_cnnrnn_loader(horizon, '0710', 'kfold', pm, isReg=True,
                                                              root_path=MODEL_PATH)
                cls_cnn_s, cls_cnn_res_s = best_cnn_loader(horizon, '0710', 'kfold', pm, isReg=False,
                                                           root_path=MODEL_PATH)
                reg_cnn_s, reg_cnn_res_s = best_cnn_loader(horizon, '0710', 'kfold', pm, isReg=True,
                                                           root_path=MODEL_PATH)

            elif day.month <= 11 and day.month >= 10:
                cls_rnn_s, cls_rnn_res_s = best_cnnrnn_loader(horizon, '0912', 'kfold', pm, isReg=False,
                                                              root_path=MODEL_PATH)
                reg_rnn_s, reg_rnn_res_s = best_cnnrnn_loader(horizon, '0912', 'kfold', pm, isReg=True,
                                                              root_path=MODEL_PATH)
                cls_cnn_s, cls_cnn_res_s = best_cnn_loader(horizon, '0912', 'kfold', pm, isReg=False,
                                                           root_path=MODEL_PATH)
                reg_cnn_s, reg_cnn_res_s = best_cnn_loader(horizon, '0912', 'kfold', pm, isReg=True,
                                                           root_path=MODEL_PATH)

            else:
                cls_rnn_s, cls_rnn_res_s = best_cnnrnn_loader(horizon, '1102', 'kfold', pm, isReg=False,
                                                              root_path=MODEL_PATH)
                reg_rnn_s, reg_rnn_res_s = best_cnnrnn_loader(horizon, '1102', 'kfold', pm, isReg=True,
                                                              root_path=MODEL_PATH)
                cls_cnn_s, cls_cnn_res_s = best_cnn_loader(horizon, '1102', 'kfold', pm, isReg=False,
                                                           root_path=MODEL_PATH)
                reg_cnn_s, reg_cnn_res_s = best_cnn_loader(horizon, '1102', 'kfold', pm, isReg=True,
                                                           root_path=MODEL_PATH)

            reg_cnn_res_s['param']['model_name'] = 'cnn'
            reg_rnn_res_s['param']['model_name'] = 'rnn'

            models = [cls_rnn, reg_rnn, cls_cnn, reg_cnn, cls_rnn_s, reg_rnn_s, cls_cnn_s, reg_cnn_s]
            results = [cls_rnn_res, reg_rnn_res, cls_cnn_res, reg_cnn_res, cls_rnn_res_s, reg_rnn_res_s, cls_cnn_res_s,
                       reg_cnn_res_s]
            #         models = [cls_cnn, reg_cnn, cls_cnn_s, reg_cnn_s]
            #         results = [cls_cnn_res, reg_cnn_res, cls_cnn_res_s, reg_cnn_res_s]

            j, outputs = run_v3_test(models, day, results)

            #         print(j)

            if j > 0:
                (best_reg_output, best_cls_output, ensemble_reg_output, ensemble_cls_output, ensemble_total_return,
                 model_index) = outputs
                best_results[horizon].append(
                    {'best_reg': best_reg_output, 'best_cls': best_cls_output, 'ensemble_reg': ensemble_reg_output,
                     'ensemble_cls': ensemble_cls_output,
                     'ensemble_total': ensemble_total_return, 'model_index': model_index})

            save_data(best_results, '.', SAVE_FILE)

            day = day + timedelta(days=1)

