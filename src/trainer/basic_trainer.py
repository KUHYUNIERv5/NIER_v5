#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/10/26 1:45 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @`File`      : basic_trainer_dev2.py
# @Software  : PyCharm

from typing import Union, List, Iterable
# from collections.abc import Iterable
from abc import ABC
import os

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim, nn

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

from copy import deepcopy
import logging
import numpy as np

from ..dataset import NIERDataset
from ..utils import AverageMeter, set_random_seed, concatenate
from ..models import DoubleInceptionModel, SingleInceptionModel, SingleInceptionCRNN, DoubleInceptionCRNN, \
    DoubleInceptionModel_v2, SingleInceptionModel_v2, SingleInceptionCRNN_v2, \
    DoubleInceptionCRNN_v2


class BasicTrainer(ABC):
    """Trainer base class for cross validation of NIER models."""

    def __init__(self, pm_type: str, model_name: str = 'CNN', model_type: str = "single", model_ver: str = 'v2',
                 scheduler: str = "MultiStepLR", lr_milestones=None, optimizer_name: str = "SGD",
                 objective_name: str = "CrossEntropyLoss", n_epochs: int = 1, dropout: float = 0.,
                 device: str = "cpu", name: str = "NIER_R5", is_reg=False, log_path: str = "../../log",
                 seed: int = 999, log_flag: bool = True):
        """
        Trainer parameters
        @param pm_type: pm type
        @param model_name: name of model ("CNN or RNN")
        @param model_type: means the number of layers ("double" or "single")
        @param model_ver:
        @param scheduler:
        @param lr_milestones:
        @param optimizer_name:
        @param objective_name:
        @param n_epochs:
        @param dropout:
        @param device:
        @param name:
        @param is_reg:
        @param log_path:
        @param seed:
        @param log_flag:
        """
        super(BasicTrainer).__init__()
        if lr_milestones is None:
            lr_milestones = [40]
        set_random_seed(seed)  # for reproduceability

        assert model_ver in ['v1', 'v2'], f'bad numeric type: {model_ver}'
        self.pm_type = pm_type
        self.model_name = model_name
        self.model_type = model_type
        self.model_ver = model_ver
        self.scheduler = scheduler
        self.lr_milestones = lr_milestones
        self.is_reg = is_reg
        self.optimizer_name = optimizer_name
        self.objective_name = objective_name
        self.n_epochs = n_epochs
        self.dropout = dropout
        self.device = device
        self.optimizer = None
        self.seed = seed
        self.objective = None
        if not os.path.exists(log_path):
            os.mkdir(log_path)

        self.log_path = f'{log_path}/{name}.log'
        self.log_flag = log_flag

        self._reset_history()
        self.is_custom_obj = False
        self.logger = self._get_logger()

    def __history__(self):
        return self.history

    def _reset_history(self):
        self.history = dict(
            kfold_results=[],
            #             model_list=[],
            best_model_list=[],
            best_fold_idx=0,
        )

    def _evaluation(self, y_list, pred_list):
        """
        **필요시 변경해야 함(현재는 단기 팀 세팅 따름)**
        :param y_list: true values
        :param pred_list: predicted values
        :return: object (accuracy, hit, pod, far, f1)
        """
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

    def _get_logger(self):
        loglevel = logging.INFO
        l = logging.getLogger(self.log_path)
        if not getattr(l, 'handler_set', None):
            l.setLevel(loglevel)
            h = logging.StreamHandler()
            f = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            h.setFormatter(f)
            l.addHandler(h)
            l.setLevel(loglevel)
            l.handler_set = True
        return l

    def optimizer_setup(self, parameters, optimizer_name=None, **optimizer_args):
        if optimizer_name is None:
            optimizer_name = self.optimizer_name
        if hasattr(optim, optimizer_name):
            self.optimizer = getattr(optim, optimizer_name)(parameters, **optimizer_args)
        else:
            raise AttributeError(f"{optimizer_name} is not a valid attribute in torch.optim.")

    def objective_setup(self, objective_name=None, **objective_args):
        if self.is_reg:
            self.objective = nn.MSELoss(**objective_args)
        elif not self.is_reg:
            self.objective = nn.CrossEntropyLoss(**objective_args)
        else:
            self.objective = getattr(nn, objective_name)(**objective_args)

    def setup(self, net_parameters, optimizer_args, objective_args):
        # print(optimizer_args)
        self.optimizer_setup(net_parameters, optimizer_name=self.optimizer_name, **optimizer_args)
        self.objective_setup(objective_name=self.objective_name, **objective_args)

    def cross_validate(self, dataset: NIERDataset, model_args: dict, optimizer_args: dict, objective_args: dict,
                       param_args: dict, K=3, batch_size=64, ):
        splits = KFold(n_splits=K, shuffle=True, random_state=self.seed)
        self.lag = model_args['lag']
        self.horizon = param_args['horizon']
        self.sampling = param_args['sampling']
        kfold_models = []
        kfold_results = []

        if dataset.numeric_scenario == 4 and dataset.horizon > 3 and self.model_ver == 'v2':
            is_point_added = True
        else:
            is_point_added = False

        setting = f"Device {self.device}| Horizon: {self.horizon} | {self.model_name} | Layer: {self.model_type} | Reg: {self.is_reg} | Lag: {self.lag} | Sampl: {self.sampling}"
        if self.log_flag:
            self.logger.info(setting)

        net = None
        mean, scale, thresholds = dataset.mean, dataset.scale, dataset.threshold_dict[self.pm_type]
        # kfold cross validation
        for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
            cv_run = f"Device {self.device} | Horizon: {self.horizon} | Fold {fold + 1}"
            if self.log_flag:
                self.logger.info(cv_run)
            if self.model_ver == 'v1':
                if self.model_name == 'CNN':
                    if self.model_type == 'single':
                        net = SingleInceptionModel(dropout=self.dropout, reg=self.is_reg, **model_args)
                    elif self.model_type == 'double':
                        net = DoubleInceptionModel(dropout=self.dropout, reg=self.is_reg, **model_args)
                    else:
                        net = DoubleInceptionModel(dropout=self.dropout, reg=self.is_reg, **model_args)

                if self.model_name == 'RNN':
                    if self.model_type == 'single':
                        net = SingleInceptionCRNN(dropout=self.dropout, reg=self.is_reg, rnn_type='GRU', **model_args)
                    elif self.model_type == 'double':
                        net = DoubleInceptionCRNN(dropout=self.dropout, reg=self.is_reg, rnn_type='GRU', **model_args)
                    else:
                        net = DoubleInceptionCRNN(dropout=self.dropout, reg=self.is_reg, rnn_type='GRU', **model_args)
            else:
                if self.model_name == 'CNN':
                    if self.model_type == 'single':
                        net = SingleInceptionModel_v2(dropout=self.dropout, reg=self.is_reg, added_point=is_point_added,
                                                      **model_args)
                    elif self.model_type == 'double':
                        net = DoubleInceptionModel_v2(dropout=self.dropout, reg=self.is_reg, added_point=is_point_added,
                                                      **model_args)
                    else:
                        net = DoubleInceptionModel_v2(dropout=self.dropout, reg=self.is_reg, added_point=is_point_added,
                                                      **model_args)

                if self.model_name == 'RNN':
                    if self.model_type == 'single':
                        net = SingleInceptionCRNN_v2(dropout=self.dropout, reg=self.is_reg, rnn_type='GRU',
                                                     added_point=is_point_added, **model_args)
                    elif self.model_type == 'double':
                        net = DoubleInceptionCRNN_v2(dropout=self.dropout, reg=self.is_reg, rnn_type='GRU',
                                                     added_point=is_point_added, **model_args)
                    else:
                        net = DoubleInceptionCRNN_v2(dropout=self.dropout, reg=self.is_reg, rnn_type='GRU',
                                                     added_point=is_point_added, **model_args)

            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
            val_loader = DataLoader(dataset, batch_size=1000, sampler=val_sampler)

            model_weights, best_model_weights, return_dict = self.train(train_loader, val_loader, scale, mean,
                                                                        thresholds, net, optimizer_args=optimizer_args,
                                                                        objective_args=objective_args, is_esv=False)

            if best_model_weights is None:
                best_model_weights = model_weights
            kfold_models.append(best_model_weights)
            kfold_results.append(return_dict)

        if self.model_ver == 'v1':
            if self.model_name == 'CNN':
                if self.model_type == 'single':
                    net = SingleInceptionModel(dropout=self.dropout, reg=self.is_reg, **model_args)
                elif self.model_type == 'double':
                    net = DoubleInceptionModel(dropout=self.dropout, reg=self.is_reg, **model_args)
                else:
                    net = DoubleInceptionModel(dropout=self.dropout, reg=self.is_reg, **model_args)

            if self.model_name == 'RNN':
                if self.model_type == 'single':
                    net = SingleInceptionCRNN(dropout=self.dropout, reg=self.is_reg, rnn_type='GRU', **model_args)
                elif self.model_type == 'double':
                    net = DoubleInceptionCRNN(dropout=self.dropout, reg=self.is_reg, rnn_type='GRU', **model_args)
                else:
                    net = DoubleInceptionCRNN(dropout=self.dropout, reg=self.is_reg, rnn_type='GRU', **model_args)
        else:
            if self.model_name == 'CNN':
                if self.model_type == 'single':
                    net = SingleInceptionModel_v2(dropout=self.dropout, reg=self.is_reg, added_point=is_point_added,
                                                  **model_args)
                elif self.model_type == 'double':
                    net = DoubleInceptionModel_v2(dropout=self.dropout, reg=self.is_reg, added_point=is_point_added,
                                                  **model_args)
                else:
                    net = DoubleInceptionModel_v2(dropout=self.dropout, reg=self.is_reg, added_point=is_point_added,
                                                  **model_args)

            if self.model_name == 'RNN':
                if self.model_type == 'single':
                    net = SingleInceptionCRNN_v2(dropout=self.dropout, reg=self.is_reg, rnn_type='GRU',
                                                 added_point=is_point_added, **model_args)
                elif self.model_type == 'double':
                    net = DoubleInceptionCRNN_v2(dropout=self.dropout, reg=self.is_reg, rnn_type='GRU',
                                                 added_point=is_point_added, **model_args)
                else:
                    net = DoubleInceptionCRNN_v2(dropout=self.dropout, reg=self.is_reg, rnn_type='GRU',
                                                 added_point=is_point_added, **model_args)

        trainloader = DataLoader(dataset, batch_size=batch_size)
        for epoch in range(self.n_epochs):
            # --------------- Train stage ---------------#
            _, _, _, _ = self._run_epoch(trainloader, net, scale, mean)

        f1_list = [result['best_f1'] for result in kfold_results]
        val_f1_score = np.mean(f1_list)

        return net, kfold_models, val_f1_score, kfold_results

    def single_train(self, train_set: NIERDataset, valid_set: NIERDataset, model_args: dict,
                     optimizer_args: dict, objective_args: dict, param_args: dict, batch_size=64, ):
        self.lag = model_args['lag']
        self.horizon = param_args['horizon']
        self.sampling = param_args['sampling']
        self.region = param_args['region']

        if train_set.numeric_scenario == 4 and train_set.horizon > 3 and self.model_ver == 'v2':
            is_point_added = True
        else:
            is_point_added = False

        setting = f"{self.region} | Device:{self.device} | Horizon:{self.horizon} | {self.model_name} | is_Reg:{self.is_reg} | Lag:{self.lag} | " \
                  f"{self.model_type} | {self.sampling} "
        if self.log_flag:
            self.logger.info(setting)

        mean, scale, thresholds = train_set.mean, train_set.scale, train_set.threshold_dict[self.pm_type]

        if self.model_ver == 'v1':
            if self.model_name == 'CNN':
                if self.model_type == 'single':
                    net = SingleInceptionModel(dropout=self.dropout, reg=self.is_reg, **model_args)
                elif self.model_type == 'double':
                    net = DoubleInceptionModel(dropout=self.dropout, reg=self.is_reg, **model_args)
                else:
                    net = DoubleInceptionModel(dropout=self.dropout, reg=self.is_reg, **model_args)

            if self.model_name == 'RNN':
                if self.model_type == 'single':
                    net = SingleInceptionCRNN(dropout=self.dropout, reg=self.is_reg, rnn_type='GRU', **model_args)
                elif self.model_type == 'double':
                    net = DoubleInceptionCRNN(dropout=self.dropout, reg=self.is_reg, rnn_type='GRU', **model_args)
                else:
                    net = DoubleInceptionCRNN(dropout=self.dropout, reg=self.is_reg, rnn_type='GRU', **model_args)
        else:
            if self.model_name == 'CNN':
                if self.model_type == 'single':
                    net = SingleInceptionModel_v2(dropout=self.dropout, reg=self.is_reg, added_point=is_point_added,
                                                  **model_args)
                elif self.model_type == 'double':
                    net = DoubleInceptionModel_v2(dropout=self.dropout, reg=self.is_reg, added_point=is_point_added,
                                                  **model_args)
                else:
                    net = DoubleInceptionModel_v2(dropout=self.dropout, reg=self.is_reg, added_point=is_point_added,
                                                  **model_args)

            if self.model_name == 'RNN':
                if self.model_type == 'single':
                    net = SingleInceptionCRNN_v2(dropout=self.dropout, reg=self.is_reg, rnn_type='GRU',
                                                 added_point=is_point_added, **model_args)
                elif self.model_type == 'double':
                    net = DoubleInceptionCRNN_v2(dropout=self.dropout, reg=self.is_reg, rnn_type='GRU',
                                                 added_point=is_point_added, **model_args)
                else:
                    net = DoubleInceptionCRNN_v2(dropout=self.dropout, reg=self.is_reg, rnn_type='GRU',
                                                 added_point=is_point_added, **model_args)

        train_loader = DataLoader(train_set, batch_size=batch_size)
        val_loader = DataLoader(valid_set, batch_size=batch_size)

        model_weights, best_model_weights, return_dict = self.train(train_loader, val_loader, scale, mean,
                                                                    thresholds, net,
                                                                    optimizer_args=optimizer_args,
                                                                    objective_args=objective_args)
        return net, model_weights, best_model_weights, return_dict

    def esv_train(self, train_set: NIERDataset, valid_sets: Iterable[NIERDataset], esv_years: Iterable[int],
                  model_args: dict, optimizer_args: dict, objective_args: dict, param_args: dict, batch_size=64, ):
        """
        esv version training function
        @param train_set: trainset to train model with
        @param valid_sets: validataion sets of esv years
        @param esv_years: list of years indicating the validation year in esv validation
        @param model_args: arguments of the model
        @param optimizer_args: arguments of the optimizer
        @param objective_args: arguments of the objective
        @param param_args: arguments including parameters for the training, etc.
        @param batch_size: size of batches in loaders
        @return:
        """
        assert self.model_ver == 'v2', f'model version except for v2 is not acceptable. current: {self.model_ver}'
        self.lag = model_args['lag']
        self.horizon = param_args['horizon']
        self.sampling = param_args['sampling']
        self.region = param_args['region']

        if train_set.numeric_scenario == 4 and train_set.horizon > 3 and self.model_ver == 'v2':
            is_point_added = True
        else:
            is_point_added = False

        setting = f"{self.region} | Device:{self.device} | Horizon:{self.horizon} | {self.model_name} | is_Reg:{self.is_reg} | Lag:{self.lag} | " \
                  f"{self.model_type} | {self.sampling} "
        if self.log_flag:
            self.logger.info(setting)

        mean, scale, thresholds = train_set.mean, train_set.scale, train_set.threshold_dict[self.pm_type]
        net = None

        if self.model_name == 'CNN':
            if self.model_type == 'single':
                net = SingleInceptionModel_v2(dropout=self.dropout, reg=self.is_reg, added_point=is_point_added,
                                              **model_args)
            elif self.model_type == 'double':
                net = DoubleInceptionModel_v2(dropout=self.dropout, reg=self.is_reg, added_point=is_point_added,
                                              **model_args)
            else:
                net = DoubleInceptionModel_v2(dropout=self.dropout, reg=self.is_reg, added_point=is_point_added,
                                              **model_args)
        elif self.model_name == 'RNN':
            if self.model_type == 'single':
                net = SingleInceptionCRNN_v2(dropout=self.dropout, reg=self.is_reg, rnn_type='GRU',
                                             added_point=is_point_added, **model_args)
            elif self.model_type == 'double':
                net = DoubleInceptionCRNN_v2(dropout=self.dropout, reg=self.is_reg, rnn_type='GRU',
                                             added_point=is_point_added, **model_args)
            else:
                net = DoubleInceptionCRNN_v2(dropout=self.dropout, reg=self.is_reg, rnn_type='GRU',
                                             added_point=is_point_added, **model_args)

        train_loader = DataLoader(train_set, batch_size=batch_size)
        valid_loaders = []
        for valid_set, esv_year in zip(valid_sets, esv_years):
            valid_loaders.append(DataLoader(valid_set, batch_size=1000))

        _, best_model_weights, return_dict = self.train(train_loader, valid_loaders, scale, mean, thresholds, net,
                                                        optimizer_args=optimizer_args, objective_args=objective_args,
                                                        is_esv=True, esv_years=esv_years)

        return net, best_model_weights, return_dict

    def _thresholding(self, array, thresholds):
        y = array.squeeze()
        y_cls = np.zeros_like(y)
        for i, threshold in enumerate(thresholds):
            y_cls[y > threshold] = i + 1
        return y_cls

    def train(self, trainloader: DataLoader, validloader: Union[DataLoader, Iterable[DataLoader]], scale: float,
              mean: float, thresholds: list, net: nn.Module, optimizer_args: dict, objective_args: dict,
              is_esv: bool = False, esv_years=None):
        """
        Implement train method that trains the given network using the train_set of dataset.

        @param trainloader: loader with training set
        @param validloader: loader or loaders with validataion set
        @param scale: scale vector for reversing standardscale
        @param mean: mean vector for reversing standardscale
        @param thresholds: thresholds for classification (pm thresholds)
        @param net: neural nets to train
        @param optimizer_args: optimizer arguments
        @param objective_args: object function arguments
        @param is_esv: flag for the esv or not
        @param esv_years: list of years for esv
        @return: Trained net, best network state_dict, results of validation sets
        """

        best_model_weights = None
        best_loss = -1111
        return_dict = dict(
            train_score_list=[],
            val_score_list=[],
            train_loss_list=[],
            val_loss_list=[],
            best_f1=0.,
            best_epoch=0,
            best_result=0.,
            best_pred=None,  # prediction lists
            y_label=None,  # label lists
            best_orig_pred=None  # original prediction lists
        )

        if is_esv:
            best_losses = {}
            best_model_weights = {}
            return_dict = dict(
                train_score_list=[],
                train_loss_list=[],
                val_loss_list={},
                val_score_list={},
                best_f1s={},
                best_epochs={},
                best_results={},
                best_preds={},  # prediction lists
                y_labels={},  # label lists
                best_orig_preds={}  # original prediction lists
            )
            for esv_year in esv_years:
                best_losses[esv_year] = -1111
                return_dict['val_loss_list'][esv_year] = []
                return_dict['val_score_list'][esv_year] = []

        self.setup(net.parameters(), optimizer_args, objective_args)
        net.to(self.device)

        scheduler = None
        if not self.scheduler is None:
            scheduler = getattr(optim.lr_scheduler, self.scheduler)(self.optimizer, self.lr_milestones)
        # self.logger.info(f"Device {self.device} start training")

        for epoch in range(self.n_epochs):
            # --------------- Train stage ---------------#
            train_orig_pred, train_pred, train_label, train_loss = self._run_epoch(trainloader, net, scale, mean)

            if self.is_reg:
                train_pred_score = self._thresholding(train_pred, thresholds)
                train_label_score = self._thresholding(train_label, thresholds)
            else:
                train_pred_score = train_pred
                train_label_score = train_label

            train_score = self._evaluation(train_label_score, train_pred_score)

            if self.is_reg:
                # print(self.objective)
                train_score['RMSE'] = np.sqrt(train_loss)

            return_dict['train_score_list'].append(train_score)
            return_dict['train_loss_list'].append(train_loss)
            # --------------- Train stage end ---------------#

            # --------------- Validation stage ---------------#
            if is_esv:
                msg_obj = {
                    'val_loss': {},
                    'val_score': {}
                }

                val_loss_avg = 0

                for vl, esv_year in zip(validloader, esv_years):
                    val_orig_pred, val_pred, val_label, val_loss = self._run_epoch(vl, net, scale, mean, is_train=False)
                    val_loss_avg += val_loss

                    if self.is_reg:
                        val_pred_score = self._thresholding(val_pred, thresholds)
                        val_label_score = self._thresholding(val_label, thresholds)

                    else:
                        val_pred_score = val_pred
                        val_label_score = val_label

                    val_score = self._evaluation(val_label_score, val_pred_score)
                    if self.is_reg:
                        val_score['RMSE'] = np.sqrt(val_loss)

                    msg_obj['val_loss'][esv_year] = val_loss
                    msg_obj['val_score'][esv_year] = val_score['f1']
                    return_dict['val_loss_list'][esv_year].append(val_loss)
                    return_dict['val_score_list'][esv_year].append(val_score['f1'])

                    if val_score['f1'] > best_losses[esv_year] and epoch > 0:
                        best_losses[esv_year] = val_score['f1']
                        return_dict['y_labels'][esv_year] = val_label
                        return_dict['best_preds'][esv_year] = val_pred
                        return_dict['best_orig_preds'][esv_year] = val_orig_pred
                        return_dict['best_results'][esv_year] = val_score
                        return_dict['best_epochs'][esv_year] = epoch
                        return_dict['best_f1s'][esv_year] = val_score['f1']
                        best_model_weights[esv_year] = deepcopy(net.cpu().state_dict())
                        net.to(self.device)
                        message = f'Update Best model with esv {esv_year} in epoch {epoch}/{self.n_epochs}'
                        if self.log_flag:
                            self.logger.info(message)

                val_loss_avg = val_loss_avg / len(esv_years)

                message = f"{self.device} Epoch {epoch}/{self.n_epochs} | train loss: {train_loss} | valid losses: {msg_obj['val_loss']}\n" \
                          f"\t\t\t\ttrain f1 score: {train_score['f1']} | valid f1 score: {msg_obj['val_score']}"
                if self.log_flag:
                    self.logger.info(message)

            else:
                val_orig_pred, val_pred, val_label, val_loss = self._run_epoch(validloader, net, scale, mean,
                                                                               is_train=False)

                # val_loss_avg = val_loss

                if self.is_reg:
                    val_pred_score = self._thresholding(val_pred, thresholds)
                    val_label_score = self._thresholding(val_label, thresholds)
                else:
                    val_pred_score = val_pred
                    val_label_score = val_label

                val_score = self._evaluation(val_label_score, val_pred_score)
                if self.is_reg:
                    val_score['RMSE'] = np.sqrt(val_loss)

                return_dict['val_score_list'].append(val_score)
                return_dict['val_loss_list'].append(val_loss)

                message = f"{self.device} Epoch {epoch}/{self.n_epochs} | train loss: {train_loss} | valid loss: {val_loss}\n" \
                          f"\t\t\t\ttrain f1 score: {train_score['f1']} | valid f1 score: {val_score['f1']}"
                if self.log_flag:
                    self.logger.info(message)

                if val_score['f1'] > best_loss and epoch > 0:
                    best_loss = val_score['f1']
                    return_dict['y_label'] = val_label
                    return_dict['best_pred'] = val_pred
                    return_dict['best_orig_pred'] = val_orig_pred
                    return_dict['best_result'] = val_score
                    return_dict['best_epoch'] = epoch
                    return_dict['best_f1'] = val_score['f1']
                    best_model_weights = deepcopy(net.cpu().state_dict())
                    net.to(self.device)
                    message = f'Update Best models in epoch {epoch}/{self.n_epochs}'
                    if self.log_flag:
                        self.logger.info(message)

            # --------------- Validation stage end ---------------#

            if not self.scheduler is None:
                if self.scheduler == 'ReduceLROnPlateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

        # check esv return dicts
        if is_esv:
            for esv_year in esv_years:
                if esv_year not in list(best_model_weights.keys()):
                    print(f"Warning: No best model for year {esv_year}")
                    best_model_weights[esv_year] = deepcopy(net.cpu().state_dict())
                    net.to(self.device)

        return net.cpu().state_dict(), best_model_weights, return_dict

    def _run_epoch(self, dataloader, net, scale, mean, is_train=True):
        if is_train:
            net.train()
        else:
            net.eval()

        loss_ = AverageMeter()
        pred_list = None
        y_list = None
        original_pred_list = None

        for i, batch in enumerate(dataloader):  # obs, fnl, numeric, y(scaled), y(original), y(classification)
            # handle batch data
            x_obs = batch[0].to(self.device).float()
            x_obs = x_obs.permute(0, 2, 1).contiguous()
            x_fnl = batch[1].to(self.device).float()
            x_fnl = x_fnl.permute(0, 2, 1).contiguous()
            x_num = batch[2].to(self.device).float()
            point_num = None
            if batch[-1].shape[-1] != 1 and self.model_ver == 'v2':
                point_num = batch[-1].to(self.device).float()

            # print('debug', x_obs.shape, x_fnl.shape, x_num.shape, point_num.shape)

            if self.is_reg:
                y = batch[3].to(self.device).float()
                orig_y = batch[4].to(self.device).float()
            else:
                y = batch[5].to(self.device).long()

            x_pred = net(x_obs, x_fnl, x_num, point_num)

            if self.is_reg:
                x_pred = x_pred.squeeze(-1) if x_pred.shape[0] == 1 else x_pred.squeeze()
                original_pred_list = x_pred.detach().cpu().numpy()
                y_list = concatenate(y_list, orig_y.detach().cpu().numpy())
                pred_list = concatenate(pred_list, x_pred.detach().cpu().numpy() * scale + mean)
            else:
                preds = x_pred.argmax(dim=1)
                y_list = concatenate(y_list, y.detach().cpu().numpy())
                pred_list = concatenate(pred_list, preds.detach().cpu().numpy())
                original_pred_list = concatenate(original_pred_list, x_pred.detach().cpu().numpy(), axis=0)

            loss = self.objective(x_pred, y)
            loss_.update(loss.item(), x_obs.shape[0])

            # update
            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return original_pred_list, pred_list, y_list, loss_.avg

    def test(self, testset: NIERDataset, net, batch_size, optimizer_args: dict = None, objective_args: dict = None):
        """
        Implement test method that evaluates the test_set of dataset on the given network.
        """
        scale, mean, threshold = testset.scale, testset.mean, testset.threshold_dict[self.pm_type]
        test_loader = DataLoader(testset, batch_size=1000, shuffle=False)

        if self.optimizer is None or self.objective is None:
            self.setup(net.parameters(), optimizer_args, objective_args)
        net.to(self.device)
        net.eval()

        with torch.no_grad():
            test_orig_pred, test_pred, test_label, test_loss = self._run_epoch(test_loader, net, scale, mean,
                                                                               is_train=False)

        if self.is_reg:
            test_pred_score = self._thresholding(test_pred, threshold)
            test_label_score = self._thresholding(test_label, threshold)
        else:
            test_pred_score = test_pred
            test_label_score = test_label

        test_score = self._evaluation(test_label_score, test_pred_score)

        if self.is_reg:
            test_score['RMSE'] = np.sqrt(test_loss)

        return test_score, test_orig_pred, test_pred_score, test_label_score
