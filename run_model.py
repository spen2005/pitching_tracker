import torch
import torch.nn as nn
import numpy as np

# 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 加载训练好的模型参数
def load_model():
    model = SimpleNN(input_size=19, hidden_size=64, output_size=3)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    return model

# 输入数据预处理
def preprocess_input(vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, timestamp, x_prime, y_prime):
    input_data = np.array([vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, timestamp, x_prime, y_prime], dtype=np.float32)
    return torch.tensor(input_data)

# 用户交互
def user_input():
    print("Please enter the following inputs:")
    vec1 = float(input("vec1[0]: "))
    vec2 = float(input("vec1[1]: "))
    vec3 = float(input("vec2[0]: "))
    vec4 = float(input("vec2[1]: "))
    vec5 = float(input("vec3[0]: "))
    vec6 = float(input("vec3[1]: "))
    vec7 = float(input("vec4[0]: "))
    vec8 = float(input("vec4[1]: "))
    timestamp = float(input("Timestamp: "))
    x_prime = float(input("x_prime: "))
    y_prime = float(input("y_prime: "))
    return vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, timestamp, x_prime, y_prime

# 模型预测
def predict(model, input_data):
    with torch.no_grad():
        output = model(input_data.unsqueeze(0))
    return output.squeeze().cpu().numpy()

if __name__ == "__main__":
    # 加载模型
    model = load_model()

    # 用户交互
    vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, timestamp, x_prime, y_prime = user_input()

    # 输入数据预处理
    input_data = preprocess_input(vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, timestamp, x_prime, y_prime)

    # 模型预测
    x_pred, y_pred, z_pred = predict(model, input_data)

    # 输出预测结果
    print(f"Predicted x: {x_pred}")
    print(f"Predicted y: {y_pred}")
    print(f"Predicted z: {z_pred}")

