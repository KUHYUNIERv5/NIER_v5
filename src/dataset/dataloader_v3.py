#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2023/04/13 10:30 AM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : dataloader_v3.py
# @Software  : PyCharm

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils import load_data

class V3Dataset(Dataset):
    def __init__(self,
                 predict_location_id,
                 predict_pm,
                 rm_region,
                 period_version,
                 data_path,
                 lag: int = 1,
                 horizon: int = 4,
                 ):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        horizon_day = torch.tensor([0.]).float()
        if torch.is_tensor(item):
            item = item.tolist()
        index = self.idx_list[item]


