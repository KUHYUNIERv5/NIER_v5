## 미세먼지중기예측 5차년도 과제

### 001. 데이터 생성

실행 파일 : prepare_dataset.py

prepare_dataset.py 파일에서 region_lists에 학습에 필요한 권역 리스트를 넣은 후 실행하여 데이터를 생성함

수정된 argument 정보
/data_folder/settings.yaml 파일에 최종 선정된 최적 권역 제거 그룹 및 입력기간 설정 정보가 저장됨
 - predict_region : 예측 대상 권역
 - representative_region: 예측 대상 권역이 따르는 대표 권역
 - remove_region : 예측 대상 권역 중 제거 그룹 개수 (0, 1, 2, 3)
 - rmgroup_file : 각 예측 대상 권역 별 제거 그룹 목록
 - 입력기간:
   - p1: 17-21
   - p2: 18-21
   - p3: 19-21
   - p4: 20-21


### 002. 데이터 로드 후 학습

실행 파일 : trainer_validation.py

### 참고사항 (수정됨)

[//]: # ( - 데이터 생성과 데이터 학습시에 사용되는 .pkl 명명 규칙)

[//]: # ( - 예측권역_학습기간_제거권역그룹개수 &#40; ex : R4_62_20190101_20211231_rmgroup_0 &#41;)

[//]: # ( - 예측권역, 학습기간, 제거권역그룹을 모두 저장하여 학습시 사용하기 위함)

 - 각 세팅 별로 고유의 uid를 지정하도록 변경함
 - 데이터 저장 규칙:
   - 'root_dir' 로 설정된 directory에 예측권역 별로 결과를 저장
   - 각 세팅(outer hyperparameter, inner hyperparameter 포함)마다 고유 uuid를 지정하여 각각 따로 파일을 저장함
   - {root_dir}/results 폴더에는 esv 결과, cv 결과, testset에 대한 결과가 저장됨
   - {root_dir}/models 폴더에는 esv에 따른 각 모델 state_dict 및 세팅에 맞는 network를 저장함
