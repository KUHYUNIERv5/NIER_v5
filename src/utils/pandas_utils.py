#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2023/03/29 4:05 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : pandas_utils.py
# @Software  : PyCharm

import pandas as pd

def stringify_list(df, col):
    df[col] = [','.join(map(str, l)) for l in df[col]]
    return df