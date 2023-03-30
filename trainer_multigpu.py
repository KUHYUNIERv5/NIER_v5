import torch.multiprocessing as mp
import os

import torch
from itertools import product
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.model_selection import ParameterGrid
from collections import OrderedDict
from src.trainer import BasicTrainer
from src.dataset import NIERDataset
from src.utils import save_data
import argparse
from tqdm.auto import tqdm
import gc

import warnings
warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser(description='retrain arg')

parser.add_argument('--region', help='input region', default='R4_62')
parser.add_argument('--pm_type', help='input pm type', default='PM10')
parser.add_argument('--reg', action='store_true', help='input clsreg type')
parser.add_argument('--epoch', type=int, default=50, help='input epoch')
parser.add_argument('--num_scen', type=int, default=0, help='input numeric scenario')
parser.add_argument('--model_ver', default='v1', help='model version')
# parser.add_argument('--lr', required=True, type=float, default=0.0001, help='input running device')

### Change or add arguments here
parser.add_argument('--root_dir', default='./NIER_v5/results/', help='save directory')
parser.add_argument('--data_path', type=str, default='./NIER_v5/data_folder/', help="dataset directory")

parser.add_argument('--period', default ='19_to_21', help='period name') # 17_to_21, 18_to_21, 19_to_21, 20_to_21, 17_to_19, 19_to_21_test_ver
parser.add_argument('--rm_region', type=int, default=0, help="remove region num : 0, 1, 2, 3")
parser.add_argument('--rm_group_file', type=str, default="./NIER_v5/data_folder/height_region_list.csv")

parser.add_argument('--gpu_list', nargs='+', type=int, default=[4,5])

args = parser.parse_args()

# args configure
predict_location_id = args.region
pm_types = ["PM10", "PM25"]

date_dict = {
    "17_to_21": [20170101, 20211231],
    "18_to_21": [20180101, 20211231],
    "19_to_21": [20190101, 20211231],
    "20_to_21": [20200101, 20211231],
    "17_to_19": [20170101, 20191231],
    "19_to_21_test_ver" : [20200101, 20211231]
}

start_date, until_date = date_dict[args.period][0], date_dict[args.period][1]

exp_name = args.region + "_" + str(start_date) + '_' + str(until_date) + '_rmgroup_' + str(args.rm_region) # ex) R4_62_20190101_20211231_rmgroup_1

print("exp_name : ", exp_name)

data_path = os.path.join(args.data_path, args.region)

rm_region = args.rm_region
is_reg = False
n_epochs = args.epoch
numeric_scenario = 0
model_ver = args.model_ver
save_path = os.path.join(args.root_dir, args.region, exp_name)
gpu_idx_list = args.gpu_list
csv_name = args.rm_group_file


# Dataset_args
# lag = [1,2,3,4,5,6,7]
lag = [3]
device='cuda:0'
pca_dim = dict(
        obs=256,
        fnl=512,
        wrf=128,
        cmaq=512,
        numeric=512
    )
# samplings = ['original','oversampling']
samplings = ['original']
numeric_data_handling = 'normal'

# Trainer_dict
dropout = 0.1
# model_names = ['RNN','CNN']
model_name = 'RNN'
model_types = ['single']
# model_types = ['single', 'double']
log_flag=False

# save dir
if is_reg:
    run_type = 'regression'
else:
    run_type = 'classification'

if model_ver == 'v1':
    numeric_scenario = 0
    numeric_data_handling = 'single'

if numeric_scenario == 0 and model_ver == 'v1':
    num_setting = 'r4'
elif numeric_scenario == 3:
    num_setting = 'r5'
else:
    num_setting = 'r6'

def multi_GPU(params):
    gpu_idx = semaphore.pop()
    device = 'cuda:%d' % gpu_idx if torch.cuda.is_available() else 'cpu'
    
    score_dict = dict()
    model_dict = dict()
    for idx, input_param in enumerate(tqdm(params['param_grid'])):
        
        dataset_args = input_param[0]
        trainer_dict = input_param[1]
        if trainer_dict['is_reg']:
            trainer_dict['objective_name'] = 'MSELoss'
            lr = 0.0001
        else:
            trainer_dict['objective_name'] = 'CrossEntropyLoss'
            lr = 0.0001
        numeric_lag = 1 if dataset_args['numeric_scenario'] == 0 else dataset_args['numeric_scenario']
        train_val_dict = dict(
            model_args=dict(
                obs_dim=dataset_args['pca_dim']['obs'],
                fnl_dim=dataset_args['pca_dim']['fnl'],
                num_dim=dataset_args['pca_dim'][dataset_args['numeric_type']],
                lag=dataset_args['lag'],
            ),
            optimizer_args={
                'lr': lr,
                'momentum': 0.9,
                'weight_decay': 1e-5,
                'nesterov': True
            },
            param_args={
                'horizon': dataset_args['horizon'],
                'sampling': dataset_args['sampling'],
                'region': dataset_args['predict_location_id'],
            },
            objective_args={},
            batch_size=64
        )

        trainset = NIERDataset(**dataset_args)
        dataset_args['data_type'] = 'test'
        validset = NIERDataset(**dataset_args)

        trainer_dict['device'] = device

        trainer = BasicTrainer(**trainer_dict)
        net, model, best_model, return_dict = trainer.single_train(train_set=trainset, valid_set=validset,
                                                                   **train_val_dict)
        """
        SAVE format

        region: 예측한 권역 id (R4_59~R4_77)
        pm: 예측 PM (PM10, PM2.5)
        horizon: 예측일 (3 ~ 6)
        lag: 모델 과거 입력일
        sampling: 샘플링 종류 (original, oversampling)
        num_scen: WRF+CMAQ load 방식
        model_ver: 모델 버젼 (v1:이전, v2:수정 후)
        model: 모델 (CNN:1DCNN, RNN:CNN-LSTM)
        c_or_r: 문제 종류 (classification, regression)
        layer: 모델 레이어 수 (single, double)
        run_epoch: 학습 epoch
        best_epoch: best fold 최적 epoch (+1)
        lr: 학습율
        best_valid_scores: best valid scores
        {'accuracy':, 'hit':, 'pod':, 'far':, 'f1':,'RMSE':(regression only)}
        all_train_scores: 모든 epoch train scores
        all_valid_scores: 모든 epoch valid scores 
        train_loss: 모든 epoch train loss
        valid_loss: 모든 epoch validation loss
        best_val_pred_orig: best epoch validation 예측결과 실제 값
        best_val_pred: best epoch validation 예측결과 thresholding
        y_label: validation set label
        """

        run_info = dict(
            region = dataset_args['predict_location_id'],
            pm = dataset_args['predict_pm'],
            horizon = dataset_args['horizon'],
            lag = dataset_args['lag'],
            sampling = dataset_args['sampling'],
            num_scen = dataset_args['numeric_scenario'],
            model_ver = trainer_dict['model_ver'],
            model = trainer_dict['model_name'],
            c_or_r = trainer_dict['is_reg'],
            layer = trainer_dict['model_type'],
            run_epoch = trainer_dict['n_epochs'],
            best_epoch = return_dict['best_epoch'], #0부터 count
            lr = lr,
        )

        score_dict[idx] = dict(
            run_info = run_info,
            best_valid_scores = return_dict['best_score'],
            all_train_scores = return_dict['train_score_list'],
            all_valid_scores = return_dict['val_score_list'],
            train_loss = return_dict['train_loss_list'],
            valid_loss = return_dict['val_loss_list'],
            best_val_pred_orig = return_dict['best_orig_pred'],
            best_val_pred = return_dict['best_pred'],
            y_label = return_dict['y_label']
        )

        model_dict[idx] = dict(
            run_info = run_info,
            best_model = best_model,
        )

        semaphore.append(gpu_idx)

        return_score_dict[dataset_args['horizon']] = {'save_dict': score_dict,}
        return_model_dict[dataset_args['horizon']] = {'model_dict': model_dict,}

# start = time.time()
for pm_type in tqdm(pm_types):
    param_list = []
    # for horizon in range(3, 7):
    for horizon in [3, 5]:
        dataset_args = dict(
            predict_location_id=[predict_location_id],
            predict_pm=[pm_type],
            shuffle=[False],
            data_path=[data_path],
            data_type=['train'],
            pca_dim=[pca_dim],
            lag=lag,  # 예측에 사용할 lag 길이
            numeric_type=['numeric'],
            exp_name=[exp_name],
            horizon=[horizon],  # 예측하려는 horizon
            timepoint_day=[4],  # 하루에 수집되는 데이터 포인트 수 (default: 4) fixed
            interval=[1],  # 예측에 사용될 interval fixed
            max_lag=[7],  # 최대 lag 길이 (4일 ~ 1일)
            max_horizon=[6],  # 최대 horizon 길이
            rm_region=[rm_region],
            numeric_data_handling=[numeric_data_handling],  # normal: 바뀐 모델, single: 4차년도 모델
            numeric_scenario=[numeric_scenario],
            sampling=samplings,  # oversampling, normal
            serial_y=[False],
            start_date=[start_date],
            until_date=[until_date],
        )

        trainer_dict = dict(
            pm_type=[pm_type],
            model_name=[model_name],
            model_type=model_types,
            model_ver=[model_ver],
            scheduler=["MultiStepLR"],
            lr_milestones=[[40]],
            optimizer_name=["SGD"],
            objective_name=["MSELoss"],  # CrossEntropyLoss MSELoss
            n_epochs=[n_epochs],
            dropout=[dropout],
            device=[device],
            batch_first=[True],
            name=["NIER_R5"],
            is_reg=[is_reg],
            return_best=[False],
            log_path=["./log"],
            seed=[999],
            log_flag = [log_flag],
        )

        dataset_params = ParameterGrid(OrderedDict(dataset_args))
        trainer_params = ParameterGrid(OrderedDict(trainer_dict))
        input_params = list(product(dataset_params, trainer_params))
        param_list.append({'param_grid': input_params})
    if __name__ == "__main__":
        
        manager = mp.Manager()
        semaphore = manager.list(gpu_idx_list)
        return_score_dict = manager.dict()
        return_model_dict = manager.dict()
        pool = mp.Pool(processes=2)
        pool.map(multi_GPU, param_list)
        pool.close()
        pool.join()
        
        experiment = predict_location_id + '_' + pm_type
        save_dir = os.path.join(save_path, experiment)
        print(f"save data : {save_dir}")
        save_data(return_score_dict.values(), save_dir+'/scores', f'{num_setting}_{predict_location_id}_{pm_type}_{run_type}_{model_name}_score_result.pkl')
        save_data(return_model_dict.values(), save_dir+'/models', f'{num_setting}_{predict_location_id}_{pm_type}_{run_type}_{model_name}_grid_models.pkl')
# end = time.time()
# print(f"{end - start:.5f} sec")