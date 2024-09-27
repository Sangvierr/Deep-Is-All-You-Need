import numpy as np

def step_activation(x):
    return np.where(x > 0, 1, 0)

class XOR:
    def __init__(self):
        self.W_hidden = np.array([[1.0, 1.0], [1.0, 1.0]])
        self.b_hidden = np.array([-1.5, -0.5])
        self.W_output = np.array([-1.0, 1.0])
        self.b_output = np.array([-0.5])
        
    def forward(self, x):
        # 입력값 -> 은닉층
        z1 = np.dot(x, self.W_hidden) + self.b_hidden
        hidden_output = step_activation(z1)
        
        # 은닉층 -> 출력층
        z2 = np.dot(hidden_output, self.W_output) + self.b_output
        output = step_activation(z2)
        
        return output
    

if __name__ == "__main__":
    # XOR 연산
    xor = XOR()

    # 네 가지 입력
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # 각 경우에 대해 forward 연산 수행
    for input in inputs:
        print(f"Input: {input}, Output: {xor.forward(input)}")