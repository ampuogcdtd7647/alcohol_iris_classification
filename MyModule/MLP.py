import torch
import torch.nn as nn

# 定义 MLP 模型
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

batch_size = 8
time_steps = 4
input_tensor = torch.randn(batch_size, time_steps)
input_dim = input_tensor.shape[1]
hidden_dim = 64

model = SimpleMLP(input_dim, hidden_dim)

output = model(input_tensor)
print(output.shape)

