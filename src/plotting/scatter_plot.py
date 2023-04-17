#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2023/04/03 12:58 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : scatter_plot.py
# @Software  : PyCharm

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from statannotations.Annotator import Annotator
import matplotlib.patheffects as path_effects
from matplotlib.offsetbox import AnchoredText
from scipy.stats import ttest_ind

processed_data_path='/workspace/local/src/datagen/ver_4th/db_save/'
data_dir = '/workspace/local/results/v2_ku_phase1'
result_df = pd.read_excel(os.path.join(data_dir, 'total_run_result.xlsx') ,index_col=0)
co2_result_df = pd.read_excel(os.path.join(data_dir, 'co2_run_result.xlsx') ,index_col=0)
result_df['num_scenario'] = result_df['num_scenario'].replace(['r4', 'r5', 'r6'], ['FN', 'F123', 'F123N'])
co2_result_df['co2'] = co2_result_df['co2'].map({True: 'with CO2', False: 'w.o CO2'})


def _scatter_phase1(region, horizon_data, cmaq_data=None, pm='PM10', horizon=4, ):
    pm_map = dict(
        PM10='PM 10',
        PM25='PM 2.5'
    )
    # mpl.rc('figure', figsize=[8, 8])
    # mpl.rc('lines', markersize=8)

    mpl.rc('xtick', labelsize=13)
    mpl.rc('axes', labelsize=15)
    cm = plt.cm.get_cmap('jet')
    fig, ax = plt.subplots(figsize=[15, 12])

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('FAR')
    ax.set_ylabel('POD')
    ax.set_title(f"{region} {pm_map[pm]} POD and FAR of horizon {horizon}")
    pod_avg_list = []
    far_avg_list = []
    test_data = horizon_data[horizon_data['num_scenario'] == 'FN']
    # r4 pod, far 평균 계산
    pod_avg_list.append(test_data['pod'].mean())
    far_avg_list.append(test_data['far'].mean())
    # r4 best f1 annotate
    idx = test_data['f1'].argmax()
    r4_max_loc = test_data.iloc[idx]
    #     f1 = ax.scatter(x=test_data['far'], y=test_data['pod'], marker='1', vmin=0, vmax=1, label='r4', c=test_data.f1, cmap=cm, s=90, lw=1.5)
    f1 = ax.scatter(x=test_data['far'], y=test_data['pod'], marker='1', vmin=0, vmax=1, label='FN',
                    color=sns.color_palette('bright')[0], s=90, lw=1.5)
    test_data = horizon_data[horizon_data['num_scenario'] == 'F123']
    # r5 pod, far 평균 계산
    pod_avg_list.append(test_data['pod'].mean())
    far_avg_list.append(test_data['far'].mean())
    # r5 best f1 annotate
    idx = test_data['f1'].argmax()
    r5_max_loc = test_data.iloc[idx]
    #     f2 = ax.scatter(x=test_data['far'], y=test_data['pod'], marker='+', vmin=0, vmax=1, label='r5', c=test_data.f1, cmap=cm, s=90, lw=1.5)
    f2 = ax.scatter(x=test_data['far'], y=test_data['pod'], marker='+', vmin=0, vmax=1, label='F123',
                    color=sns.color_palette('bright')[1], s=90, lw=1.5)
    test_data = horizon_data[horizon_data['num_scenario'] == 'F123N']
    # r6 pod, far 평균 계산
    pod_avg_list.append(test_data['pod'].mean())
    far_avg_list.append(test_data['far'].mean())
    # r6 best f1 annotate
    idx = test_data['f1'].argmax()
    r6_max_loc = test_data.iloc[idx]
    #     f3 = ax.scatter(x=test_data['far'], y=test_data['pod'], marker='x', vmin=0, vmax=1, label='r6', c=test_data.f1, cmap=cm, s=80, lw=1.5)
    f3 = ax.scatter(x=test_data['far'], y=test_data['pod'], marker='x', vmin=0, vmax=1, label='F123N',
                    color=sns.color_palette('bright')[2], s=80, lw=1.5)
    pod_avg_list.append(horizon_data['pod'].mean())
    far_avg_list.append(horizon_data['far'].mean())
    f1_max_x = [r4_max_loc.far, r5_max_loc.far, r6_max_loc.far]
    f1_max_y = [r4_max_loc.pod, r5_max_loc.pod, r6_max_loc.pod]
    ax.scatter(x=f1_max_x, y=f1_max_y, marker='.', color='red', s=30, lw=1.5)
    ax.annotate(f'FN', xy=(r4_max_loc.far, r4_max_loc.pod),
                fontsize=14)
    ax.annotate(f'F123', xy=(r5_max_loc.far, r5_max_loc.pod),
                fontsize=14)
    ax.annotate(f'F123N', xy=(r6_max_loc.far, r6_max_loc.pod),
                fontsize=14)
    textstr = f'FN f1 max:{r4_max_loc.f1:.3f}\n' \
              f'F123 f1 max:{r5_max_loc.f1:.3f}\n' \
              f'F123N f1 max:{r6_max_loc.f1:.3f}\n'
    if not cmaq_data is None:
        textstr = textstr + f'CMAQ f1 score:{cmaq_data.f1.values[0]:.3f}'
    textbox = AnchoredText(textstr, loc='upper right', prop=dict(size=15), )
    ax.add_artist(textbox)

    if not cmaq_data is None:
        if cmaq_data.far.values[0] == 1 and cmaq_data.pod.values[0] == 0:
            f4 = ax.scatter(x=cmaq_data.far.values[0], y=cmaq_data.pod.values[0], marker='*',
                            color=sns.color_palette('bright')[3], s=80, lw=1)

            ax.axhline(cmaq_data.pod.values[0], linestyle='--', color=sns.color_palette('bright')[3])
            ax.axvline(cmaq_data.far.values[0], linestyle='--', color=sns.color_palette('bright')[3])
        else:
            f4 = ax.scatter(x=cmaq_data.far.values[0], y=cmaq_data.pod.values[0], marker='*',
                            color=sns.color_palette('bright')[3], s=80, lw=1)
            ax.axhline(cmaq_data.pod.values[0], linestyle='--', color=sns.color_palette('bright')[3])
            ax.axvline(cmaq_data.far.values[0], linestyle='--', color=sns.color_palette('bright')[3])


    f5 = ax.scatter(x=far_avg_list, y=pod_avg_list, marker='*', color=sns.color_palette('bright')[2], s=80, lw=1)
    ax.axhline(pod_avg_list[-1], linestyle='--', color=sns.color_palette('bright')[2])
    ax.axvline(far_avg_list[-1], linestyle='--', color=sns.color_palette('bright')[2])



    # if cmaq_data.far.values[0] == 1 and cmaq_data.pod.values[0] == 0:
    #     f4 = ax.scatter(x=cmaq_data.far.values[0], y=cmaq_data.pod.values[0], marker='*',
    #                     color=sns.color_palette('bright')[3], s=80, lw=1)
    #     f5 = ax.scatter(x=far_avg_list, y=pod_avg_list, marker='*', color=sns.color_palette('bright')[2], s=80, lw=1)
    #     ax.axhline(cmaq_data.pod.values[0], linestyle='--', color=sns.color_palette('bright')[3])
    #     ax.axvline(cmaq_data.far.values[0], linestyle='--', color=sns.color_palette('bright')[3])
    #     ax.axhline(pod_avg_list[-1], linestyle='--', color=sns.color_palette('bright')[2])
    #     ax.axvline(far_avg_list[-1], linestyle='--', color=sns.color_palette('bright')[2])
    # else:
    #     f4 = ax.scatter(x=cmaq_data.far.values[0], y=cmaq_data.pod.values[0], marker='*',
    #                     color=sns.color_palette('bright')[3], s=80, lw=1)
    #     ax.axhline(cmaq_data.pod.values[0], linestyle='--', color=sns.color_palette('bright')[3])
    #     ax.axvline(cmaq_data.far.values[0], linestyle='--', color=sns.color_palette('bright')[3])
    #     plt.colorbar(f1, ax=ax, label='f1 score')

    # img = ax.imshow(test_data.to_numpy())
    # legend1 = plt.legend([f1], ["Main legend"], fontsize=12, loc=3, bbox_to_anchor=(0,0.1,0,0), frameon=False)
    # legend2 = plt.legend((f2, f3), ('sublegend 1', 'sublegend 2'), fontsize=9,
    #                  loc=3, bbox_to_anchor=(0.05,0,0,0), frameon=False)

    plt.legend(handles=[f1, f2, f3], title='num scenario', loc='center right', bbox_to_anchor=(1.0, 0.78),
               prop=dict(size=15))
    plt.show()


def _scatter_phase1(region, horizon_data, cmaq_data=None, pm='PM10', horizon=4, ):
    pm_map = dict(
        PM10='PM 10',
        PM25='PM 2.5'
    )
    # mpl.rc('figure', figsize=[8, 8])
    # mpl.rc('lines', markersize=8)

    mpl.rc('xtick', labelsize=13)
    mpl.rc('axes', labelsize=15)
    cm = plt.cm.get_cmap('jet')
    fig, ax = plt.subplots(figsize=[15, 12])

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('FAR')
    ax.set_ylabel('POD')
    ax.set_title(f"{region} {pm_map[pm]} POD and FAR of horizon {horizon}")
    pod_avg_list = []
    far_avg_list = []
    test_data = horizon_data[horizon_data['num_scenario'] == 'FN']
    # r4 pod, far 평균 계산
    pod_avg_list.append(test_data['pod'].mean())
    far_avg_list.append(test_data['far'].mean())
    # r4 best f1 annotate
    idx = test_data['f1'].argmax()
    r4_max_loc = test_data.iloc[idx]
    #     f1 = ax.scatter(x=test_data['far'], y=test_data['pod'], marker='1', vmin=0, vmax=1, label='r4', c=test_data.f1, cmap=cm, s=90, lw=1.5)
    f1 = ax.scatter(x=test_data['far'], y=test_data['pod'], marker='1', vmin=0, vmax=1, label='FN',
                    color=sns.color_palette('bright')[0], s=90, lw=1.5)
    test_data = horizon_data[horizon_data['num_scenario'] == 'F123']
    # r5 pod, far 평균 계산
    pod_avg_list.append(test_data['pod'].mean())
    far_avg_list.append(test_data['far'].mean())
    # r5 best f1 annotate
    idx = test_data['f1'].argmax()
    r5_max_loc = test_data.iloc[idx]
    #     f2 = ax.scatter(x=test_data['far'], y=test_data['pod'], marker='+', vmin=0, vmax=1, label='r5', c=test_data.f1, cmap=cm, s=90, lw=1.5)
    f2 = ax.scatter(x=test_data['far'], y=test_data['pod'], marker='+', vmin=0, vmax=1, label='F123',
                    color=sns.color_palette('bright')[1], s=90, lw=1.5)
    test_data = horizon_data[horizon_data['num_scenario'] == 'F123N']
    # r6 pod, far 평균 계산
    pod_avg_list.append(test_data['pod'].mean())
    far_avg_list.append(test_data['far'].mean())
    # r6 best f1 annotate
    idx = test_data['f1'].argmax()
    r6_max_loc = test_data.iloc[idx]
    #     f3 = ax.scatter(x=test_data['far'], y=test_data['pod'], marker='x', vmin=0, vmax=1, label='r6', c=test_data.f1, cmap=cm, s=80, lw=1.5)
    f3 = ax.scatter(x=test_data['far'], y=test_data['pod'], marker='x', vmin=0, vmax=1, label='F123N',
                    color=sns.color_palette('bright')[2], s=80, lw=1.5)
    pod_avg_list.append(horizon_data['pod'].mean())
    far_avg_list.append(horizon_data['far'].mean())
    f1_max_x = [r4_max_loc.far, r5_max_loc.far, r6_max_loc.far]
    f1_max_y = [r4_max_loc.pod, r5_max_loc.pod, r6_max_loc.pod]
    ax.scatter(x=f1_max_x, y=f1_max_y, marker='.', color='red', s=30, lw=1.5)
    ax.annotate(f'FN', xy=(r4_max_loc.far, r4_max_loc.pod),
                fontsize=14)
    ax.annotate(f'F123', xy=(r5_max_loc.far, r5_max_loc.pod),
                fontsize=14)
    ax.annotate(f'F123N', xy=(r6_max_loc.far, r6_max_loc.pod),
                fontsize=14)
    textstr = f'FN f1 max:{r4_max_loc.f1:.3f}\n' \
              f'F123 f1 max:{r5_max_loc.f1:.3f}\n' \
              f'F123N f1 max:{r6_max_loc.f1:.3f}\n'
    if not cmaq_data is None:
        textstr = textstr + f'CMAQ f1 score:{cmaq_data.f1.values[0]:.3f}'
    textbox = AnchoredText(textstr, loc='upper right', prop=dict(size=15), )
    ax.add_artist(textbox)

    if not cmaq_data is None:
        if cmaq_data.far.values[0] == 1 and cmaq_data.pod.values[0] == 0:
            f4 = ax.scatter(x=cmaq_data.far.values[0], y=cmaq_data.pod.values[0], marker='*',
                            color=sns.color_palette('bright')[3], s=80, lw=1)

            ax.axhline(cmaq_data.pod.values[0], linestyle='--', color=sns.color_palette('bright')[3])
            ax.axvline(cmaq_data.far.values[0], linestyle='--', color=sns.color_palette('bright')[3])
        else:
            f4 = ax.scatter(x=cmaq_data.far.values[0], y=cmaq_data.pod.values[0], marker='*',
                            color=sns.color_palette('bright')[3], s=80, lw=1)
            ax.axhline(cmaq_data.pod.values[0], linestyle='--', color=sns.color_palette('bright')[3])
            ax.axvline(cmaq_data.far.values[0], linestyle='--', color=sns.color_palette('bright')[3])


    f5 = ax.scatter(x=far_avg_list, y=pod_avg_list, marker='*', color=sns.color_palette('bright')[2], s=80, lw=1)
    ax.axhline(pod_avg_list[-1], linestyle='--', color=sns.color_palette('bright')[2])
    ax.axvline(far_avg_list[-1], linestyle='--', color=sns.color_palette('bright')[2])



    # if cmaq_data.far.values[0] == 1 and cmaq_data.pod.values[0] == 0:
    #     f4 = ax.scatter(x=cmaq_data.far.values[0], y=cmaq_data.pod.values[0], marker='*',
    #                     color=sns.color_palette('bright')[3], s=80, lw=1)
    #     f5 = ax.scatter(x=far_avg_list, y=pod_avg_list, marker='*', color=sns.color_palette('bright')[2], s=80, lw=1)
    #     ax.axhline(cmaq_data.pod.values[0], linestyle='--', color=sns.color_palette('bright')[3])
    #     ax.axvline(cmaq_data.far.values[0], linestyle='--', color=sns.color_palette('bright')[3])
    #     ax.axhline(pod_avg_list[-1], linestyle='--', color=sns.color_palette('bright')[2])
    #     ax.axvline(far_avg_list[-1], linestyle='--', color=sns.color_palette('bright')[2])
    # else:
    #     f4 = ax.scatter(x=cmaq_data.far.values[0], y=cmaq_data.pod.values[0], marker='*',
    #                     color=sns.color_palette('bright')[3], s=80, lw=1)
    #     ax.axhline(cmaq_data.pod.values[0], linestyle='--', color=sns.color_palette('bright')[3])
    #     ax.axvline(cmaq_data.far.values[0], linestyle='--', color=sns.color_palette('bright')[3])
    #     plt.colorbar(f1, ax=ax, label='f1 score')

    # img = ax.imshow(test_data.to_numpy())
    # legend1 = plt.legend([f1], ["Main legend"], fontsize=12, loc=3, bbox_to_anchor=(0,0.1,0,0), frameon=False)
    # legend2 = plt.legend((f2, f3), ('sublegend 1', 'sublegend 2'), fontsize=9,
    #                  loc=3, bbox_to_anchor=(0.05,0,0,0), frameon=False)

    plt.legend(handles=[f1, f2, f3], title='num scenario', loc='center right', bbox_to_anchor=(1.0, 0.78),
               prop=dict(size=15))
    plt.show()


def _scatter_phase2(region, result_df, cmaq_data=None, pm='PM10', horizon=4, check_column='lag'):
    pm_map = dict(
        PM10='PM 10',
        PM25='PM 2.5'
    )

    str_expr = f"(predict_region == '{region}') and (pm_type == '{pm}') and (horizon == {horizon})"

    cmaqdf_q = cmaq_data.query(str_expr)
    result_q = result_df.query(str_expr)

    check_columns = result_q[check_column].unique()
    check_columns = np.sort(check_columns)
    # mpl.rc('figure', figsize=[8, 8])
    # mpl.rc('lines', markersize=8)

    mpl.rc('xtick', labelsize=13)
    mpl.rc('axes', labelsize=15)
    cm = plt.cm.get_cmap('jet')
    fig, ax = plt.subplots(figsize=[15, 12])

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('FAR')
    ax.set_ylabel('POD')
    ax.set_title(f"{region} {pm_map[pm]} POD and FAR of horizon {horizon}")
    pod_avg_list = []
    far_avg_list = []
    fig_list = []
    max_loc_list = []

    marker_list = ['1', '+', '2', 'x', '3', '^', '4']
    label_list = [f'{check_column} {val}' for val in check_columns]

    for i, column_val in enumerate(check_columns):
        test_data = result_q[result_q[check_column] == column_val]
        pod_avg_list.append(test_data['val_pod'].mean())
        far_avg_list.append(test_data['val_far'].mean())
        # r4 best f1 annotate
        idx = test_data['val_f1'].argmax()
        max_loc_list.append(test_data.iloc[idx])
        # r4_max_loc = test_data.iloc[idx]
        #     f1 = ax.scatter(x=test_data['far'], y=test_data['pod'], marker='1', vmin=0, vmax=1, label='r4', c=test_data.f1, cmap=cm, s=90, lw=1.5)
        fig = ax.scatter(x=test_data['val_far'], y=test_data['val_pod'], marker=marker_list[i], vmin=0, vmax=1,
                         label=label_list[i],
                         color=sns.color_palette('bright')[i], s=90, lw=1.5)
        fig_list.append(fig)

    pod_avg_list.append(result_q['val_pod'].mean())
    far_avg_list.append(result_q['val_far'].mean())

    f1_max_far = [ml.val_far for ml in max_loc_list]
    f1_max_pod = [ml.val_pod for ml in max_loc_list]
    f1s = [ml.val_f1 for ml in max_loc_list]

    ax.scatter(x=f1_max_far, y=f1_max_pod, marker='.', color='red', s=30, lw=1.5)
    textstr = ''
    for max_far, max_pod, f1, check_col in zip(f1_max_far, f1_max_pod, f1s, check_columns):
        ax.annotate(f'{check_col}', xy=(max_far, max_pod), fontsize=14)
        textstr += f'{check_col} f1 max:{f1:.3f}\n'

    if not cmaqdf_q is None:
        textstr = textstr + f'CMAQ f1 score:{cmaqdf_q.f1.values[0]:.3f}'

    textbox = AnchoredText(textstr, loc='upper right', prop=dict(size=15), )
    ax.add_artist(textbox)

    if not cmaqdf_q is None:
        if cmaqdf_q.far.values[0] == 1 and cmaqdf_q.pod.values[0] == 0:
            f4 = ax.scatter(x=cmaqdf_q.far.values[0], y=cmaqdf_q.pod.values[0], marker='*',
                            color=sns.color_palette('bright')[3], s=80, lw=1)

            ax.axhline(cmaqdf_q.pod.values[0], linestyle='--', color=sns.color_palette('bright')[3])
            ax.axvline(cmaqdf_q.far.values[0], linestyle='--', color=sns.color_palette('bright')[3])
        else:
            f4 = ax.scatter(x=cmaqdf_q.far.values[0], y=cmaqdf_q.pod.values[0], marker='*',
                            color=sns.color_palette('bright')[3], s=80, lw=1)
            ax.axhline(cmaqdf_q.pod.values[0], linestyle='--', color=sns.color_palette('bright')[3])
            ax.axvline(cmaqdf_q.far.values[0], linestyle='--', color=sns.color_palette('bright')[3])

    f5 = ax.scatter(x=far_avg_list, y=pod_avg_list, marker='*', color=sns.color_palette('bright')[2], s=80, lw=1)
    ax.axhline(pod_avg_list[-1], linestyle='--', color=sns.color_palette('bright')[2])
    ax.axvline(far_avg_list[-1], linestyle='--', color=sns.color_palette('bright')[2])

    plt.legend(handles=fig_list, title='num scenario', loc='center right', bbox_to_anchor=(1.0, 0.78),
               prop=dict(size=15))
    plt.show()


def _co2_scatter(region, horizon_data, cmaq_data, pm='PM10', horizon=4, ):
    pm_map = dict(
        PM10='PM 10',
        PM25='PM 2.5'
    )
    # mpl.rc('figure', figsize=[8, 8])
    # mpl.rc('lines', markersize=8)

    mpl.rc('xtick', labelsize=13)
    mpl.rc('axes', labelsize=15)
    cm = plt.cm.get_cmap('jet')
    fig, ax = plt.subplots(figsize=[15, 12])

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('FAR')
    ax.set_ylabel('POD')
    ax.set_title(f"{region} {pm_map[pm]} CO2 POD and FAR of horizon {horizon}")
    pod_avg_list = []
    far_avg_list = []
    test_data = horizon_data[horizon_data['co2'] == 'w.o CO2']
    # r4 pod, far 평균 계산
    pod_avg_list.append(test_data['pod'].mean())
    far_avg_list.append(test_data['far'].mean())
    # r4 best f1 annotate
    idx = test_data['f1'].argmax()
    r4_max_loc = test_data.iloc[idx]
    #     f1 = ax.scatter(x=test_data['far'], y=test_data['pod'], marker='1', vmin=0, vmax=1, label='r4', c=test_data.f1, cmap=cm, s=90, lw=1.5)
    f1 = ax.scatter(x=test_data['far'], y=test_data['pod'], marker='1', vmin=0, vmax=1, label='w.o CO2',
                    color=sns.color_palette('bright')[0], s=90, lw=1.5)
    test_data = horizon_data[horizon_data['co2'] == 'with CO2']
    # r5 pod, far 평균 계산
    pod_avg_list.append(test_data['pod'].mean())
    far_avg_list.append(test_data['far'].mean())
    # r5 best f1 annotate
    idx = test_data['f1'].argmax()
    r5_max_loc = test_data.iloc[idx]
    #     f2 = ax.scatter(x=test_data['far'], y=test_data['pod'], marker='+', vmin=0, vmax=1, label='r5', c=test_data.f1, cmap=cm, s=90, lw=1.5)
    f2 = ax.scatter(x=test_data['far'], y=test_data['pod'], marker='+', vmin=0, vmax=1, label='with CO2',
                    color=sns.color_palette('bright')[1], s=90, lw=1.5)

    f1_max_far = [r4_max_loc.far, r5_max_loc.far]
    f1_max_pod = [r4_max_loc.pod, r5_max_loc.pod]
    ax.scatter(x=f1_max_far, y=f1_max_pod, marker='.', color=sns.color_palette('bright')[7], s=30, lw=1.5)
    ax.annotate(f'no CO2', xy=(r4_max_loc.far, r4_max_loc.pod),
                fontsize=15)
    ax.annotate(f'CO2', xy=(r5_max_loc.far, r5_max_loc.pod),
                fontsize=15)

    textstr = f'w.o co2 f1 max:{r4_max_loc.f1:.3f}\n' \
              f'with co2 f1 max:{r5_max_loc.f1:.3f}\n' \
              f'CMAQ f1 score:{cmaq_data.f1.values[0]:.3f}'
    textbox = AnchoredText(textstr, loc='upper right', prop=dict(size=15), )
    ax.add_artist(textbox)
    if cmaq_data.far.values[0] == 1 and cmaq_data.pod.values[0] == 0:
        f4 = ax.scatter(x=cmaq_data.far.values[0], y=cmaq_data.pod.values[0], marker='*',
                        color=sns.color_palette('bright')[3], s=80, lw=1)
        f5 = ax.scatter(x=far_avg_list, y=pod_avg_list, marker='*', color=sns.color_palette('bright')[2], s=80, lw=1)
        ax.axhline(cmaq_data.pod.values[0], linestyle='--', color=sns.color_palette('bright')[3])
        ax.axvline(cmaq_data.far.values[0], linestyle='--', color=sns.color_palette('bright')[3])
        ax.axhline(pod_avg_list[-1], linestyle='--', color=sns.color_palette('bright')[2])
        ax.axvline(far_avg_list[-1], linestyle='--', color=sns.color_palette('bright')[2])
    else:
        f4 = ax.scatter(x=cmaq_data.far.values[0], y=cmaq_data.pod.values[0], marker='*',
                        color=sns.color_palette('bright')[3], s=80, lw=1)
        ax.axhline(cmaq_data.pod.values[0], linestyle='--', color=sns.color_palette('bright')[3])
        ax.axvline(cmaq_data.far.values[0], linestyle='--', color=sns.color_palette('bright')[3])
    #     plt.colorbar(f1, ax=ax, label='f1 score')

    # img = ax.imshow(test_data.to_numpy())
    # legend1 = plt.legend([f1], ["Main legend"], fontsize=12, loc=3, bbox_to_anchor=(0,0.1,0,0), frameon=False)
    # legend2 = plt.legend((f2, f3), ('sublegend 1', 'sublegend 2'), fontsize=9,
    #                  loc=3, bbox_to_anchor=(0.05,0,0,0), frameon=False)

    plt.legend(handles=[f1, f2], title='co2', loc='center left', bbox_to_anchor=(-0.2, 0.9), prop=dict(size=15))
    plt.show()


def add_median_labels(ax, fmt='.3f'):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='right', va='center',
                       fontweight='bold', color='white')
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])


def ttest_annotate(result_data, horizon, alternative='less'):
    fn = list(result_data[(result_data['horizon'] == horizon) & (result_data['num_scenario'] == 'FN')].f1)
    f123 = list(result_data[(result_data['horizon'] == horizon) & (result_data['num_scenario'] == 'F123')].f1)
    f123n = list(result_data[(result_data['horizon'] == horizon) & (result_data['num_scenario'] == 'F123N')].f1)
    _, p1 = ttest_ind(a=fn, b=f123, equal_var=False, alternative=alternative)
    _, p2 = ttest_ind(a=fn, b=f123n, equal_var=False, alternative=alternative)
    return p1, p2


def co2_ttest_annotate(result_data, horizon, alternative='less'):
    wo_co2 = list(result_data[(result_data['horizon'] == horizon) & (result_data['co2'] == 'w.o CO2')].f1)
    w_co2 = list(result_data[(result_data['horizon'] == horizon) & (result_data['co2'] == 'with CO2')].f1)
    _, p1 = ttest_ind(a=wo_co2, b=w_co2, equal_var=False, alternative=alternative)
    return p1


def boxplot(result_data, region, pm='PM10', horizon_list=[3, 4, 5, 6]):
    result_data = result_data[(result_data['region'] == region) & (result_data['pm'] == pm) &
                              (result_data['horizon'].isin(horizon_list))]
    # annotate y cordinate
    y = result_data['f1'].max()
    len_horizon = len(horizon_list)
    plt.figure(figsize=(5 * len_horizon, 6))
    if len_horizon == 2:
        plt.ylim([0, 1.])
    if len_horizon == 1:
        plt.ylim([0, 1.])
    else:
        plt.ylim([0, 1.2])
    box_plot = sns.boxplot(x="horizon", y="f1", hue='num_scenario', data=result_data, hue_order=['FN', 'F123', 'F123N'],
                           showmeans=True,
                           meanprops={"marker": "+",
                                      "markeredgecolor": "black",
                                      "markersize": "10"})
    #     X = np.repeat(np.atleast_2d(np.arange(len_horizon)),2, axis=0)+ np.array([[-.27],[.27]])
    #     print(X)
    #     box_plot.plot(X.flatten(), [y+0.02,y+0.02, y+0.02,y+0.02], 'ro', zorder=4)
    add_median_labels(box_plot)
    # horizon per annotate
    if len(horizon_list) == 1:
        for horizon in horizon_list:
            if horizon == 3:
                p1, p2 = ttest_annotate(result_data, horizon)
                print('horizon3 pval ', p1, p2)
                x1, x2, x3 = -0.27, 0, 0.27
                plt.plot([x1, x1, x2, x2], [y + 0.02, y + 0.02 + 0.02, y + 0.02 + 0.02, y + 0.02], lw=1.5, c='black')
                plt.plot([x1, x1, x3, x3], [y + 0.08, y + 0.08 + 0.02, y + 0.08 + 0.02, y + 0.08], lw=1.5, c='black')
                plt.text((x1 + x2) * .5, y + 0.02 + 0.02, f"p={round(p1, 4)}", ha='center', va='bottom', color='black')
                if p2 < 1e-4:
                    plt.text((x1 + x3) * .5, y + 0.08 + 0.02, f"p≤1e-4", ha='center', va='bottom', color='black')
                else:
                    plt.text((x1 + x3) * .5, y + 0.08 + 0.02, f"p={round(p2, 4)}", ha='center', va='bottom',
                             color='black')
            if horizon == 5:
                p1, p2 = ttest_annotate(result_data, horizon)
                print('horizon5 pval ', p1, p2)
                x1, x2, x3 = -0.27, 0, 0.27
                plt.plot([x1, x1, x2, x2], [y + 0.02, y + 0.02 + 0.02, y + 0.02 + 0.02, y + 0.02], lw=1.5, c='black')
                plt.plot([x1, x1, x3, x3], [y + 0.08, y + 0.08 + 0.02, y + 0.08 + 0.02, y + 0.08], lw=1.5, c='black')
                plt.text((x1 + x2) * .5, y + 0.02 + 0.02, f"p={round(p1, 4)}", ha='center', va='bottom', color='black')
                if p2 < 1e-4:
                    plt.text((x1 + x3) * .5, y + 0.08 + 0.02, f"p≤1e-4", ha='center', va='bottom', color='black')
                else:
                    plt.text((x1 + x3) * .5, y + 0.08 + 0.02, f"p={round(p2, 4)}", ha='center', va='bottom',
                             color='black')
    if len(horizon_list) == 2:
        for horizon in horizon_list:
            if horizon == 3:
                p1, p2 = ttest_annotate(result_data, horizon)
                print('horizon3 pval ', p1, p2)
                x1, x2, x3 = -0.27, 0, 0.27
                plt.plot([x1, x1, x2, x2], [y + 0.02, y + 0.02 + 0.02, y + 0.02 + 0.02, y + 0.02], lw=1.5, c='black')
                plt.plot([x1, x1, x3, x3], [y + 0.08, y + 0.08 + 0.02, y + 0.08 + 0.02, y + 0.08], lw=1.5, c='black')
                plt.text((x1 + x2) * .5, y + 0.02 + 0.02, f"p={round(p1, 4)}", ha='center', va='bottom', color='black')
                if p2 < 1e-4:
                    plt.text((x1 + x3) * .5, y + 0.08 + 0.02, f"p≤1e-4", ha='center', va='bottom', color='black')
                else:
                    plt.text((x1 + x3) * .5, y + 0.08 + 0.02, f"p={round(p2, 4)}", ha='center', va='bottom',
                             color='black')
            if horizon == 5:
                p1, p2 = ttest_annotate(result_data, horizon)
                print('horizon5 pval ', p1, p2)
                x1, x2, x3 = 0.73, 1, 1.27
                plt.plot([x1, x1, x2, x2], [y + 0.02, y + 0.02 + 0.02, y + 0.02 + 0.02, y + 0.02], lw=1.5, c='black')
                plt.plot([x1, x1, x3, x3], [y + 0.08, y + 0.08 + 0.02, y + 0.08 + 0.02, y + 0.08], lw=1.5, c='black')
                plt.text((x1 + x2) * .5, y + 0.02 + 0.02, f"p={round(p1, 4)}", ha='center', va='bottom', color='black')
                if p2 < 1e-4:
                    plt.text((x1 + x3) * .5, y + 0.08 + 0.02, f"p≤1e-4", ha='center', va='bottom', color='black')
                else:
                    plt.text((x1 + x3) * .5, y + 0.08 + 0.02, f"p={round(p2, 4)}", ha='center', va='bottom',
                             color='black')
    if len(horizon_list) == 4:
        for horizon in horizon_list:
            if horizon == 3:
                p1, p2 = ttest_annotate(result_data, horizon)
                print('horizon3 pval ', p1, p2)
                x1, x2, x3 = -0.27, 0, 0.27
                plt.plot([x1, x1, x2, x2], [y + 0.02, y + 0.02 + 0.02, y + 0.02 + 0.02, y + 0.02], lw=1.5, c='black')
                plt.plot([x1, x1, x3, x3], [y + 0.08, y + 0.08 + 0.02, y + 0.08 + 0.02, y + 0.08], lw=1.5, c='black')
                if p1 < 1e-4:
                    plt.text((x1 + x2) * .5, y + 0.02 + 0.02, f"p≤1e-4", ha='center', va='bottom', color='black')
                else:
                    plt.text((x1 + x2) * .5, y + 0.02 + 0.02, f"p={round(p1, 4)}", ha='center', va='bottom',
                             color='black')
                if p2 < 1e-4:
                    plt.text((x1 + x3) * .5, y + 0.08 + 0.02, f"p≤1e-4", ha='center', va='bottom', color='black')
                else:
                    plt.text((x1 + x3) * .5, y + 0.08 + 0.02, f"p={round(p2, 4)}", ha='center', va='bottom',
                             color='black')
            if horizon == 4:
                p1, p2 = ttest_annotate(result_data, horizon)
                print('horizon4 pval ', p1, p2)
                x1, x2, x3 = 0.73, 1, 1.27
                plt.plot([x1, x1, x2, x2], [y + 0.02, y + 0.02 + 0.02, y + 0.02 + 0.02, y + 0.02], lw=1.5, c='black')
                plt.plot([x1, x1, x3, x3], [y + 0.08, y + 0.08 + 0.02, y + 0.08 + 0.02, y + 0.08], lw=1.5, c='black')
                if p1 < 1e-4:
                    plt.text((x1 + x2) * .5, y + 0.02 + 0.02, f"p≤1e-4", ha='center', va='bottom', color='black')
                else:
                    plt.text((x1 + x2) * .5, y + 0.02 + 0.02, f"p={round(p1, 4)}", ha='center', va='bottom',
                             color='black')
                if p2 < 1e-4:
                    plt.text((x1 + x3) * .5, y + 0.08 + 0.02, f"p≤1e-4", ha='center', va='bottom', color='black')
                else:
                    plt.text((x1 + x3) * .5, y + 0.08 + 0.02, f"p={round(p2, 4)}", ha='center', va='bottom',
                             color='black')
            if horizon == 5:
                p1, p2 = ttest_annotate(result_data, horizon)
                print('horizon5 pval ', p1, p2)
                x1, x2, x3 = 1.73, 2, 2.27
                plt.plot([x1, x1, x2, x2], [y + 0.02, y + 0.02 + 0.02, y + 0.02 + 0.02, y + 0.02], lw=1.5, c='black')
                plt.plot([x1, x1, x3, x3], [y + 0.08, y + 0.08 + 0.02, y + 0.08 + 0.02, y + 0.08], lw=1.5, c='black')
                if p1 < 1e-4:
                    plt.text((x1 + x2) * .5, y + 0.02 + 0.02, f"p≤1e-4", ha='center', va='bottom', color='black')
                else:
                    plt.text((x1 + x2) * .5, y + 0.02 + 0.02, f"p={round(p1, 4)}", ha='center', va='bottom',
                             color='black')
                if p2 < 1e-4:
                    plt.text((x1 + x3) * .5, y + 0.08 + 0.02, f"p≤1e-4", ha='center', va='bottom', color='black')
                else:
                    plt.text((x1 + x3) * .5, y + 0.08 + 0.02, f"p={round(p2, 4)}", ha='center', va='bottom',
                             color='black')
            if horizon == 6:
                p1, p2 = ttest_annotate(result_data, horizon)
                print('horizon6 pval ', p1, p2)
                x1, x2, x3 = 2.73, 3, 3.27
                plt.plot([x1, x1, x2, x2], [y + 0.02, y + 0.02 + 0.02, y + 0.02 + 0.02, y + 0.02], lw=1.5, c='black')
                plt.plot([x1, x1, x3, x3], [y + 0.08, y + 0.08 + 0.02, y + 0.08 + 0.02, y + 0.08], lw=1.5, c='black')
                if p1 < 1e-4:
                    plt.text((x1 + x2) * .5, y + 0.02 + 0.02, f"p≤1e-4", ha='center', va='bottom', color='black')
                else:
                    plt.text((x1 + x2) * .5, y + 0.02 + 0.02, f"p={round(p1, 4)}", ha='center', va='bottom',
                             color='black')
                if p2 < 1e-4:
                    plt.text((x1 + x3) * .5, y + 0.08 + 0.02, f"p≤1e-4", ha='center', va='bottom', color='black')
                else:
                    plt.text((x1 + x3) * .5, y + 0.08 + 0.02, f"p={round(p2, 4)}", ha='center', va='bottom',
                             color='black')

    plt.ylabel("F1", size=14)
    plt.xlabel("Horizon", size=14)
    plt.title(f"{region} {pm} f1 score box plot", size=10)
    if len_horizon == 4:
        legen_loc = (-0.1, 1.0)
    elif len_horizon == 1:
        legen_loc = (-0.45, 1.0)
    else:
        legen_loc = (-0.2, 1.0)
    plt.legend(loc='upper left', bbox_to_anchor=legen_loc)
    plt.show()


#     plt.savefig("grouped_boxplot_Seaborn_boxplot_Python.png")

def co2_boxplot(result_data, region, pm='PM10', horizon_list=[3, 4, 5, 6]):
    sns.set(style="white", palette="bright", color_codes=True)
    result_data = result_data[(result_data['region'] == region) & (result_data['pm'] == pm) &
                              (result_data['horizon'].isin(horizon_list))]
    y = result_data['f1'].max()
    len_horizon = len(horizon_list)
    plt.figure(figsize=(8, 6,))
    plt.ylim([0, 1.2])
    box_plot = sns.boxplot(y="f1", x="horizon", hue='co2', data=result_data, hue_order=['w.o CO2', 'with CO2'],
                           palette="bright", showmeans=True,
                           meanprops={"marker": "+", "markeredgecolor": "black", "markersize": "10"})

    add_median_labels(box_plot)

    if len(horizon_list) == 4:
        for horizon in horizon_list:
            if horizon == 3:
                p = co2_ttest_annotate(result_data, horizon)
                print('horizon3 pval ', p)
                x1, x2, = -0.2, 0.2,
                plt.plot([x1, x1, x2, x2], [y + 0.02, y + 0.02 + 0.02, y + 0.02 + 0.02, y + 0.02], lw=1.5, c='black')
                plt.text((x1 + x2) * .5, y + 0.02 + 0.02, f"p={round(p, 4)}", ha='center', va='bottom', color='black')
            if horizon == 4:
                p = co2_ttest_annotate(result_data, horizon)
                print('horizon5 pval ', p)
                x1, x2 = 0.8, 1.2
                plt.plot([x1, x1, x2, x2], [y + 0.02, y + 0.02 + 0.02, y + 0.02 + 0.02, y + 0.02], lw=1.5, c='black')
                plt.text((x1 + x2) * .5, y + 0.02 + 0.02, f"p={round(p, 4)}", ha='center', va='bottom', color='black')
            if horizon == 5:
                p = co2_ttest_annotate(result_data, horizon)
                print('horizon5 pval ', p)
                x1, x2 = 1.8, 2.2
                plt.plot([x1, x1, x2, x2], [y + 0.02, y + 0.02 + 0.02, y + 0.02 + 0.02, y + 0.02], lw=1.5, c='black')
                plt.text((x1 + x2) * .5, y + 0.02 + 0.02, f"p={round(p, 4)}", ha='center', va='bottom', color='black')
            if horizon == 6:
                p = co2_ttest_annotate(result_data, horizon)
                print('horizon5 pval ', p)
                x1, x2 = 2.8, 3.2
                plt.plot([x1, x1, x2, x2], [y + 0.02, y + 0.02 + 0.02, y + 0.02 + 0.02, y + 0.02], lw=1.5, c='black')
                plt.text((x1 + x2) * .5, y + 0.02 + 0.02, f"p={round(p, 4)}", ha='center', va='bottom', color='black')

    plt.xlabel("F1", size=14)
    plt.ylabel("Horizon", size=14)
    plt.title(f"{region} {pm} co2 f1 score box plot", size=10)
    plt.legend(title="CO2", loc='upper left', bbox_to_anchor=(-0.3, 1.0))
    plt.show()

def scatter_plotting_v1(region, data, cmaq_data=None, pm='PM10', horizon_list=[3, 4, 5, 6], ):
    for horizon in horizon_list:
        horizon_score_data = data[data.horizon == horizon]
        if not cmaq_data is None:
            horizon_cmaq_data = cmaq_data[cmaq_data.horizon == horizon]
        else:
            horizon_cmaq_data = None
        _scatter_phase1(region, horizon_score_data, horizon_cmaq_data, pm, horizon)

def scatter_plotting(region, data, cmaq_data=None, pm='PM10', horizon_list=[3, 4, 5, 6], check_column='lag'):
    for horizon in horizon_list:
        _scatter_phase2(region, data, cmaq_data, pm=pm, horizon=horizon, check_column=check_column)