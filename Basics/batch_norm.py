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

# 배치 정규화 함수
def batch_norm(x, gamma=1, beta=0, eps=1e-8):
    '''
    x: 입력값
    gamma: 스케일 파라미터
    beta: 시프트 파라미터
    '''
    
    mean = np.mean(x)
    variance = np.var(x)
    x_normalized = (x - mean) / np.sqrt(variance + eps)
    return gamma * x_normalized + beta, mean

# Simple Neural Network
class simpleNeuralNetwork:
    def __init__(self, w1_init, w2_init, learning_rate=0.1, use_batch_norm=False):
        self.w1 = w1_init
        self.w2 = w2_init
        self.lr = learning_rate
        self.use_batch_norm = use_batch_norm
        
    def forward(self, x):
        # 입력층 -> 은닉층
        self.z1 = np.dot(self.w1, x)
        self.h = sigmoid(self.z1)
        
        if self.use_batch_norm:
            self.h, self.h_mean = batch_norm(self.h)  # 배치 정규화 적용
        else:
            self.h_mean = np.mean(self.h)
        
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
            hidden_means = []
            for epoch in range(num_epochs):
                # 순전파
                y_pred = self.forward(x)
                
                # 손실 계산
                loss = self.compute_loss(y, y_pred)
                loss_list.append(np.mean(loss))
                
                # 은닉층의 출력값의 평균 저장
                hidden_means.append(self.h_mean)
                
                #print(f'Epoch {epoch+1}/{num_epochs} - Loss: {np.mean(loss)}')
                
                # 역전파
                self.backward(x, y)
            
            return loss_list, hidden_means
            
        
        
if __name__ == '__main__':
    # 문제에서 주어진 가중치
    w1_init = np.array([[0.1, 0.4, 0.5], [0.2, 0.3, 0.6]])
    w2_init = np.array([0.7, 0.9])

    # 입력 데이터
    x = np.array([0.5, 0.2, 0.8])
    y = 1

    # 배치 정규화 미적용
    nn_without_bn = simpleNeuralNetwork(w1_init, w2_init, learning_rate=0.1, use_batch_norm=False)
    loss_without_bn, means_without_bn = nn_without_bn.train(x, y, num_epochs=10000)

    # 배치 정규화 적용
    nn_with_bn = simpleNeuralNetwork(w1_init, w2_init, learning_rate=0.1, use_batch_norm=True)
    loss_with_bn, means_with_bn = nn_with_bn.train(x, y, num_epochs=10000)

    # 1행 2열로 플랏 구성
    plt.figure(figsize=(12, 6))
    
    # 배치 정규화 미적용 - 은닉층 출력의 평균 변화
    plt.subplot(1, 2, 1)
    plt.plot(means_without_bn, label="Without Batch Normalization - Mean")
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Hidden Layer Output (Without Batch Normalization)')
    plt.legend()
    
    # 배치 정규화 적용 - 은닉층 출력의 평균 변화 
    plt.subplot(1, 2, 2)
    plt.plot(means_with_bn, label="With Batch Normalization - Mean")
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Hidden Layer Output (With Batch Normalization)')
    plt.legend()

    plt.tight_layout()
    plt.show()