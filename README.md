## 미세먼지중기예측 5차년도 과제

### 001. 데이터 생성

실행 파일 : data_gen.ipynb

data_gen.ipynb 에서 src.dataset > MakeNIERDataset 을 로드하여 실행

MakeNIERDataset(reset_db=False, save_processed_data=True, run_pca=True, start_date=20190101, until_date=20211231, test_date=20210101, predict_region=region, remove_region=rm_num, preprocess_root='data_folder/', root_dir='data_folder/', rmgroup_file='../NIER_v5/data_folder/height_region_list.csv')

추가된 argument 정보
 - predict_region : 예측 대상 권역
 - remove_region : 예측 대상 권역 중 제거 그룹 개수 (0, 1, 2, 3)
 - rmgroup_file : 각 예측 대상 권역 별 제거 그룹 목록


### 002. 데이터 로드 후 학습

실행 파일 : trainer_multigpu.py

추가된 argument 정보
 - period : 입력하고자 하는 기간
 - rm_region : 예측 대상 권역 중 제거 그룹 개수 (0, 1, 2, 3)
 - rm_group_file : 각 예측 대상 권역 별 제거 그룹 목록

### 참고사항
 - 데이터 생성과 데이터 학습시에 사용되는 .pkl 명명 규칙
 - 예측권역_학습기간_제거권역그룹개수 ( ex : R4_62_20190101_20211231_rmgroup_0 )
 - 예측권역, 학습기간, 제거권역그룹을 모두 저장하여 학습시 사용하기 위함