import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import time

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

# 训练函数
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    print_every = 10000
    model.to(device)
    start_time = time.time()
    total_iterations = 0
    iterloss = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            iterloss += loss.item()
            total_iterations += 1
            if total_iterations % print_every == 0:
                print(f"Iteration {total_iterations}, Loss: {iterloss / print_every}")
                iterloss = 0
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}, Time elapsed: {epoch_time:.2f} seconds, Estimated remaining time: {(num_epochs - epoch - 1) * epoch_time:.2f} seconds")

    # 保存模型参数
    torch.save(model.state_dict(), 'model.pth')


# 主程序
if __name__ == "__main__":
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 加载训练数据，并将其移到所选的设备上
    train_dataset = CustomDataset('train_data.csv')

    # 定义模型并将其移到所选的设备上
    input_size = len(train_dataset[0][0])
    output_size = len(train_dataset[0][1])
    hidden_size = 64
    model = SimpleNN(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 创建数据加载器，并将其移到所选的设备上
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, device)
