# PyTorch-Study
🧐 Deep Learning 모델 공부 with PyTorch

### Theme 0. PyTorch GPU Trial
- 일시 : -
- 사용한 데이터셋 : -
- 개요 : CUDA 연결 확인 및 GPU 사용 여부 확인
- 참고 : -

### Theme 1. Fashion MNIST
- 일시 : 2024.01.28
- 사용한 데이터셋 : 토치비전에 내장된 예제 데이터인 ```fashion_mnist```
- 개요 : 운동화, 셔츠, 샌들 같은 작은 이미지를 분류하는 합성곱 신경망 모델 설계
- 참고 : 딥러닝 파이토치 교과서 5장 2절의 합성곱 신경망 맛보기

### Theme 2. Stock Price Prediction
- 일시 : 2024.02.05
- 사용한 데이터셋 : 야후에서 다운 받은 스타벅스 주가 데이터
- 개요 : 종가, 시장가 등의 정보를 담은 데이터를 바탕으로 주가를 예측하는 순환 신경망 모델 설계
- 참고 : 딥러닝 파이토치 교과서 7장 5절의 LSTM

### Theme 3. Cat and Dog Classification
- 일시 : 2024.02.21
- 사용한 데이터셋 : 교재에서 제공하는 고양이, 강아지 이미지
- 개요 : 이미지 분류 모델에 전이학습(특성 추출, 파인튜닝)을 수행하여, 고양이와 강아지 분류하는 모델(ResNet) 설계 
- 참고 : 딥러닝 파이토치 교과서 5장 3절의 전이학습

### Theme 4. CNN for Image Classification
- 일시 : 2024.02.25
- 사용한 데이터셋 : 교재에서 제공하는 고양이, 강아지 이미지
- 개요 : 고양이와 강아지 분류하는 합성곱 신경망 모델(LeNet) 설계 
- 참고 : 딥러닝 파이토치 교과서 6장 3절의 이미지 분류를 위한 신경망

### Theme 5. Custom Neural Network for Tabular Data
- 일시 : 2024.02.28 & 2024.04.04
- 사용한 데이터셋 : IDA 2016 Challenge에서 진행한 APS Failure and Operational Data for Scania Trucks 데이터
- 개요 : PyTorch를 사용하여 정형데이터 처리 성능이 좋은 딥러닝 모델 설계
- 참고 : 2023-여름학기 P-SAT 방학 세미나

### Theme 6. MNIST Classification (with related techniques)
- 일시 : 2024.04.05
- 사용한 데이터셋 : torchvision MNIST 데이터셋
- 개요 : MNIST 분류를 위해 직접 Trainer를 정의하고, 다양한 테크닉을 활용해서 성능 향상 시도
- 참고 : -

### Theme 7. Waste Image Classification
- 일시 : 2024.12.18
- 사용한 데이터셋 : Kaggle ([Recyclable and Household Waste Classification](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification))
- 개요 : 분리수거 데이터셋을 활용하여 EfficientNet-V2, PyramidNet 등의 Vision 모델의 성능 개선 시도
- 참고 : -