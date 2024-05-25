import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train import CustomDataset  # 修改导入路径

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

# 测试函数
def test_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    print(f"Test Loss: {running_loss / len(test_loader)}")

if __name__ == "__main__":
    # 加载测试数据，并将其移到所选的设备上
    test_dataset = CustomDataset('../data/test_data.csv')  # 修改数据路径

    # 定义模型并将其移到所选的设备上
    input_size = len(test_dataset[0][0])
    output_size = len(test_dataset[0][1])
    hidden_size = 128
    model = SimpleNN(input_size, hidden_size, output_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 加载模型参数
    model.load_state_dict(torch.load('../models/model.pth'))  # 修改模型路径

    # 创建测试数据加载器，并将其移到所选的设备上
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 定义损失函数
    criterion = nn.MSELoss()

    # 测试模型
    test_model(model, test_loader, criterion, device)
