import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import time
import math

# 加载数据集
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data.iloc[idx, :-3].values
        label = self.data.iloc[idx, -3:].values
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# 定义神经网络模型
class ComplexNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(ComplexNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size2)
        self.fc4 = nn.Linear(hidden_size2, hidden_size1)
        self.fc5 = nn.Linear(hidden_size1, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# 加载训练好的模型参数
def load_model():
    model = ComplexNN(input_size=11, hidden_size1=128, hidden_size2=64, output_size=3)
    #gpu
    #model.load_state_dict(torch.load('../models/model_all_nt.pth'))
    #cpu
    model.load_state_dict(torch.load('../models/model_all_nt.pth', map_location=torch.device('cpu')))

    model.eval()
    return model

# 用户交互
def user_input(input_data):
    input_data = [float(num) for num in input_data]  # 将输入转换为浮点数
    normalized_input_data = []
    
    for i in range(1,5):
        input_data[i*2] -= input_data[0]
        input_data[i*2+1] -= input_data[1]

    regularization_value = math.sqrt(input_data[2]**2 + input_data[3]**2)
    for i in range(1,5):
        vec_x = input_data[i*2]
        vec_y = input_data[i*2+1]

        if regularization_value != 0:
            vec_x /= regularization_value
            vec_y /= regularization_value
        normalized_input_data.extend([vec_x, vec_y])
    timestamp = input_data[10]
    x_prime = input_data[11]
    y_prime = input_data[12]
    x_prime = x_prime - input_data[0]
    y_prime = y_prime - input_data[1]

    if regularization_value != 0:
        x_prime /= regularization_value
        y_prime /= regularization_value
    normalized_input_data.extend([timestamp, x_prime, y_prime])
    #print regularized input data
    #print("Normalized input data:")
    #print(normalized_input_data)
    return normalized_input_data

# 模型预测
def predict(model, input_data):
    with torch.no_grad():
        output = model(torch.tensor(input_data, dtype=torch.float32).unsqueeze(0))
    return output.squeeze().cpu().numpy()

if __name__ == "__main__":
    # 加载模型
    model = load_model()

    # 读取文件内容
    with open('../video/fastball_7/fastball_7-1.txt', 'r') as file:
        input_lines = file.readlines()  # 逐行读取文件内容

    predictions = []

    for line in input_lines:
        input_data = line.split()  # 按空格分割每行的数据
        input_data = user_input(input_data)
        prediction = predict(model, input_data)
        predictions.append(prediction)

    # 输出预测结果
    formatted_predictions = []

    for prediction in predictions:
        formatted_prediction = [f"{x/100.0:.3f}" for x in prediction]
        formatted_predictions.append("(" + ", ".join(formatted_prediction) + ")")

    output_string = "\n".join(formatted_predictions)  # 将列表转换为字符串

    print(output_string)
