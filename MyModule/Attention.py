import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class Attention(nn.Module):
    def __init__(self, embed_dim):
        super(Attention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # 输入形状：[batch_size, 1, 193]
        Q = self.query(x)  # Query: [batch_size, 1, 193]
        K = self.key(x)    # Key:   [batch_size, 1, 193]
        V = self.value(x)  # Value: [batch_size, 1, 193]

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(x.size(-1))
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, 1, 1]
        
        # 计算注意力输出
        output = torch.matmul(attention_weights, V)  # [batch_size, 1, 193]
        return output