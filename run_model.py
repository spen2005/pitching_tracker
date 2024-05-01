import torch
import torch.nn as nn
import math

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

# 用户交互
def user_input():
    print("Please enter the following inputs:")
    input_data = []
    for i in range(8):
        vec_x = float(input(f"vec{i+1}.x: "))
        vec_y = float(input(f"vec{i+1}.y: "))
        # 归一化
        norm = math.sqrt(vec_x**2 + vec_y**2)
        if norm != 0:
            vec_x /= norm
            vec_y /= norm
        input_data.extend([vec_x, vec_y])
    timestamp = float(input("Timestamp: "))
    x_prime = float(input("x_prime: "))
    y_prime = float(input("y_prime: "))
    # regularize
    norm = math.sqrt(x_prime**2 + y_prime**2)
    if norm != 0:
        x_prime /= norm
        y_prime /= norm
    input_data.extend([timestamp, x_prime, y_prime])
    return input_data

# 模型预测
def predict(model, input_data):
    with torch.no_grad():
        output = model(torch.tensor(input_data, dtype=torch.float32).unsqueeze(0))
    return output.squeeze().cpu().numpy()

if __name__ == "__main__":
    # 加载模型
    model = load_model()

    # 用户交互
    input_data = user_input()

    # 模型预测
    x_pred, y_pred, z_pred = predict(model, input_data)

    # 输出预测结果
    print(f"Predicted x: {x_pred}")
    print(f"Predicted y: {y_pred}")
    print(f"Predicted z: {z_pred}")
