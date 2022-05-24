----파일 설명----
## create_dataset.py
웹캠 촬영으로 손 좌표 저장 ==> 손 모델 학습 데이터 획득

## gesture_predict.py
gesture 예측 테스트

## HandTrackingModule.py
여러가지 모듈 저장

## index.html
pyscript ==> Local서버 구동(Go live 클릭)

## KCL_predict.py
손글씨 예측 테스트

## Project.py
main

## train.ipynb
gesture recognition 모델 학습

----폴더 설명----
## dataset
create_dataset.py로 학습한 데이터 저장

## figure
train.ipynb로 학습후 train_acc, val_acc 그래프 저장

## image
Local 서버에 출력할 image01.png 저장 및 기타 이미지 저장

## KCL_data
Korean Commercial Letter 목록 text

## models
model.h5 - gesture recognition 모델
Korean_commercial_letter_2350_20epoch.h5 - KCL 모델
Korean_commercial_letter_2350_20epoch_test.h5 - KCL 모델