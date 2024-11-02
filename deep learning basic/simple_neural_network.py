# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 시그모이드 함수의 미분
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# MSE
def mean_squared_error(y, y_pred):
    '''
    y, y_pred 모두 np.array 형태
    '''
    
    return 0.5 * (y-y_pred)**2

# Simple Neural Network
class simpleNeuralNetwork:
    def __init__(self, w1_init, w2_init, learning_rate=0.1):
        self.w1 = w1_init
        self.w2 = w2_init
        self.lr = learning_rate
        
    def forward(self, x):
        # 입력층 -> 은닉층
        self.z1 = np.dot(self.w1, x)
        self.h = sigmoid(self.z1)
        
        # 은닉층 -> 출력층
        self.z2 = np.dot(self.w2, self.h)
        self.y_pred = sigmoid(self.z2)
        
        return self.y_pred
    
    def compute_loss(self, y, y_pred):
        return mean_squared_error(y, y_pred)
    
    def backward(self, x, y):
        # 출력층 -> 은닉층(chain rule 적용)
        d_loss_d_y_pred = self.y_pred - y
        d_y_pred_d_z2 = sigmoid_derivative(self.z2)
        d_z2_d_w2 = self.h
        
        # 출력층의 기울기
        d_loss_d_w2 = d_loss_d_y_pred * d_y_pred_d_z2 * d_z2_d_w2

        # 은닉층 -> 입력층(chain rule 적용)
        d_z2_d_h = self.w2
        d_h_d_z1 = sigmoid_derivative(self.z1)
        d_z1_d_w1 = x
        
        # 은닉층의 기울기
        d_loss_d_w1 = np.outer((d_loss_d_y_pred * d_y_pred_d_z2 * d_z2_d_h) * d_h_d_z1, d_z1_d_w1)
        
        # 가중치 업데이트
        self.w1 -= self.lr * d_loss_d_w1
        self.w2 -= self.lr * d_loss_d_w2
        
    def train(self, x, y, num_epochs):
        loss_list = []
        for epoch in range(num_epochs):
            # 순전파
            y_pred = self.forward(x)
            
            # 손실 계산
            loss = self.compute_loss(y, y_pred)
            loss_list.append(loss)
            print(f'Epoch {epoch+1}/{num_epochs} - Loss: {np.mean(loss)}')
            
            # 역전파
            self.backward(x, y)
            
            # 가중치 업데이트 후 출력
            print(f'\tUpdated W1: \n{self.w1}')
            print(f'\tUpdated W2: \n{self.w2}\n')
            
        
        # loss 변동 그래프 출력
        plt.plot(loss_list)
        plt.show()
        
if __name__ == '__main__':
    # 문제에서 주어진 가중치
    w1_init = np.array([[0.1, 0.4, 0.5], [0.2, 0.3, 0.6]])
    w2_init = np.array([0.7, 0.9])

    # 입력 데이터
    x = np.array([0.5, 0.2, 0.8])
    y = 1

    # 신경망 초기화
    nn = simpleNeuralNetwork(w1_init, w2_init, learning_rate=0.1)

    # 10번의 Epoch 동안 훈련
    nn.train(x, y, num_epochs=10)