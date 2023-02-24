#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/11/02 4:19 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : transformer_trainer.py
# @Software  : PyCharm

import numpy as np

from .basic_trainer_dev2 import BasicTrainer
from ..models import Transformer, BERT
from ..utils import AverageMeter, easycat, set_random_seed, concatenate

from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from sklearn.model_selection import KFold


class TransformerTrainer(BasicTrainer):
    def __init__(self, **kwargs):
        super(TransformerTrainer, self).__init__(**kwargs)

    def cross_validate(self, dataset: Dataset, model_args: dict, optimizer_args: dict, objective_args: dict, K=3, batch_size=64):
        splits = KFold(n_splits=K, shuffle=True, random_state=self.seed)

        mean, scale, thresholds = dataset.mean, dataset.scale, dataset.threshold_dict[self.pm_type]

        # kfold cross validation
        for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
            print(f'Fold {fold + 1}')

            if self.model_type == 'bert':
                net = BERT(**model_args)
            else:
                net = Transformer(**model_args)

            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
            val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

            model_weights, best_model_weights, return_dict = self.train(train_loader, val_loader, scale, mean,
                                                                        thresholds, net,
                                                                        optimizer_args=optimizer_args,
                                                                        objective_args=objective_args)
            self.history['kfold_results'].append(return_dict)
            self.history['model_list'].append(model_weights)
            self.history['best_model_list'].append(best_model_weights)

        best_f1_list = [result['best_f1'] for result in self.history['kfold_results']]
        best_idx = np.argmax(best_f1_list)
        return net, self.history['model_list'][best_idx], self.history['best_model_list'][best_idx]

    def _run_epoch(self, dataloader, net, scale, mean, is_train=True):
        if is_train:
            net.train()
        else:
            net.eval()

        loss_ = AverageMeter()
        pred_list = None
        y_list = None

        for i, batch in enumerate(dataloader):  # obs, fnl, numeric, y(scaled), y(original), y(classification)
            # handle batch data
            x_obs = batch[0].to(self.device).float()
            x_obs = x_obs.permute(0, 2, 1).contiguous()
            x_fnl = batch[1].to(self.device).float()
            x_fnl = x_fnl.permute(0, 2, 1).contiguous()
            x_num = batch[2].to(self.device).float()

            if self.is_reg:
                y = batch[3].to(self.device).float()
                orig_y = batch[4].to(self.device).float()
            else:
                y = batch[5].to(self.device).long()

            x_pred = net(x_obs, x_fnl, x_num)

            if self.is_reg:
                x_pred = x_pred.squeeze()
                preds = x_pred * scale + mean
                y_list = concatenate(y_list, orig_y.detach().cpu().numpy())
                pred_list = concatenate(pred_list, preds.detach().cpu().numpy() * scale + mean)
            else:
                preds = x_pred.argmax(dim=1)
                y_list = concatenate(y_list, y.detach().cpu().numpy())
                pred_list = concatenate(pred_list, preds.detach().cpu().numpy())

            loss = self.objective(x_pred, y)
            loss_.update(loss, x_obs.shape[0])

            # update
            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return pred_list, y_list, loss_.avg
