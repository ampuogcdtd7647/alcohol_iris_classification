import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score, classification_report, accuracy_score, precision_score, recall_score
from torchvision.models import resnet18, mobilenet_v2
from PIL import Image
from torchvision.transforms import transforms
from scipy.ndimage import gaussian_filter1d
import re

# 定义加载数值数据的函数
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


# 定义 EvoSTD 模块，进行趋势和季节性分解
class EvoSTD(nn.Module):
    def __init__(self):
        super(EvoSTD, self).__init__()

    def forward(self, x):
        # 简单趋势分解（假设趋势是一个移动平均）
        trend = torch.mean(x, dim=0, keepdim=True)

        # 季节性分解（假设是原信号减去趋势）
        seasonal = x - trend

        return seasonal, trend


# 定义 InPar Attention 模块，分别处理实部和虚部的并行注意力机制
class InParAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(InParAttention, self).__init__()
        self.time_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

        # 分别为实部和虚部定义注意力机制
        self.real_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.imag_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

    def forward(self, x):
        # 时域注意力
        time_out, _ = self.time_attention(x, x, x)

        # 频域转换：对输入进行傅里叶变换
        freq_x = torch.fft.fft(x, dim=0)

        # 分离实部和虚部
        real_part = freq_x.real
        imag_part = freq_x.imag

        # 实部注意力
        real_out, _ = self.real_attention(real_part, real_part, real_part)

        # 虚部注意力
        imag_out, _ = self.imag_attention(imag_part, imag_part, imag_part)

        # 将实部和虚部的输出进行融合，可以使用相加或拼接
        freq_out = real_out + imag_out  # 可以使用相加，或者 torch.cat((real_out, imag_out), dim=-1) 拼接

        # 将时域和频域输出进行融合
        out = time_out + freq_out
        return out


# 定义最终的 Transformer 模型，结合 InPar Attention 和 EvoSTD
class TimeSeriesInParformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super(TimeSeriesInParformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)

        # InPar Attention 模块
        encoder_layers = nn.TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # EvoSTD 模块
        self.evo_std = EvoSTD()
        self.inpar_attention = InParAttention(embed_dim, num_heads, dropout)

    def forward(self, src):
        src = self.embedding(src)  # 输入数据嵌入
        src = self.pos_encoder(src)  # 添加位置编码

        # 进行 EvoSTD 分解
        seasonal, trend = self.evo_std(src)

        # 将季节性成分和趋势成分分别通过 InPar Attention
        seasonal_out = self.inpar_attention(seasonal)
        trend_out = self.inpar_attention(trend)

        # 将两者结合
        combined = seasonal_out + trend_out

        # 通过 Transformer 编码器
        output = self.transformer_encoder(combined)
        return output

# 加载数值数据并进行预处理
def sort_key(file_name):
    # 使用正则表达式将文件名分为字母部分和数字部分
    parts = re.split(r'(\d+)', file_name)
    # 处理每个部分，只有在部分是数字时才转换为整数
    return [int(part) if part.isdigit() else part for part in parts]

def load_NumericData(data_dir, max_length=150, cutoff=0.1):
    """
    从CSV文件加载数值数据并进行傅里叶变换和平滑处理。

    :param data_dir: 数据文件夹路径
    :param max_length: 信号的最大长度
    :param cutoff: 低通滤波器的截止频率
    :return: 处理后的数值数据、标签和文件名
    """
    data = []
    labels = []
    filenames = []
    
    # 获取文件夹中的所有文件，并按字母和数字顺序排序
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    sorted_files = sorted(files, key=sort_key)  # 传递文件名给sort_key进行排序

    for filename in sorted_files:
        if filename.endswith('.csv'):
            # 构造CSV文件路径
            file_path = os.path.join(data_dir, filename)
            df = pd.read_csv(file_path)
            
            # 选择需要处理的列
            column_name = 'Inner Semi-major axis (a)'

            # 检查列是否存在
            if column_name in df.columns:
                signal = df[column_name].copy()

                # 如果数据存在缺失值，可以先进行插值填充
                signal = signal.interpolate()

                # 使用3sigma原则选择异常值
                mean_signal = signal.mean()
                std_signal = signal.std()
                sigma_threshold = 3 * std_signal
                sigma_outliers = np.where((signal < mean_signal - sigma_threshold) | (signal > mean_signal + sigma_threshold))[0]

                # 将3sigma异常值替换为高斯平滑
                signal.loc[sigma_outliers] = np.nan
                signal.fillna(method='bfill', inplace=True)  # 先后向填充
                signal.fillna(method='ffill', inplace=True)  # 再前向填充
                signal_filled = gaussian_filter1d(signal, sigma=2)
                signal.loc[sigma_outliers] = signal_filled[sigma_outliers]

                # 计算二阶导数（加速度）
                second_derivative = np.gradient(np.gradient(signal))

                # 设置阈值，选择大于二阶导数标准差两倍的点为波动点
                threshold = 2 * np.std(second_derivative)
                fluctuation_indices = np.where(np.abs(second_derivative) > threshold)[0]

                # 将波动点替换为高斯平滑
                signal.loc[fluctuation_indices] = np.nan
                signal.fillna(method='bfill', inplace=True)  # 先后向填充
                signal.fillna(method='ffill', inplace=True)  # 再前向填充
                signal_filled = gaussian_filter1d(signal, sigma=2)
                signal.loc[fluctuation_indices] = signal_filled[fluctuation_indices]

                # 合并3sigma异常值和波动点索引
                combined_outliers = np.union1d(sigma_outliers, fluctuation_indices)

                # 对修正后的信号进行高斯平滑处理
                smoothed_signal = gaussian_filter1d(signal, sigma=2)
    
                # 将处理后的信号添加到数据集中
                data.append(smoothed_signal)
            
            # 标签分类：'normal' 作为标签0，其他为标签1
            label = 0 if 'normal' in filename else 1
            labels.append(label)
            filenames.append(filename)

    return data, labels, filenames

# 自定义数据集类
class MultimodalDataset(Dataset):
    def __init__(self, image_dir, numeric_data, labels, transform=None, augment_transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.augment_transform = augment_transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))], key=sort_key)
        self.numeric_data = numeric_data
        self.labels = labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        numeric = self.numeric_data[idx]
        label = self.labels[idx]

        # 图像预处理
        original_image = self.transform(image)
        if self.augment_transform:
            augmented_image = self.augment_transform(image)
            augmented_image = self.transform(augmented_image)
            extended_image = torch.cat((original_image, augmented_image), dim=0)
        else:
            extended_image = original_image

        numeric = torch.tensor(numeric, dtype=torch.float32).unsqueeze(1)  # 添加特征维度以适应 Transformer
        return extended_image, numeric, label

class MultimodalTransformerResNet(nn.Module):
    def __init__(self, num_numeric_features, embed_dim, num_heads, ff_dim, num_layers, num_classes=2):
        super(MultimodalTransformerResNet, self).__init__()
        self.transformer = TimeSeriesInParformer(input_dim=1, embed_dim=embed_dim, num_heads=num_heads,
                                                 ff_dim=ff_dim, num_layers=num_layers)
        self.resnet = resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        # 新增全连接层用于对齐 Transformer 输出的特征维度
        self.fc_transformer = nn.Linear(embed_dim, num_ftrs)  # 将Transformer输出调整为ResNet输出的大小

        self.fc1 = nn.Linear(num_ftrs + num_ftrs, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x_image, x_numeric):
        x_image = self.resnet(x_image)
        x_numeric = self.transformer(x_numeric).mean(dim=0)  # 对 Transformer 的输出进行池化

        # 调试信息
        # print(f"x_image shape: {x_image.shape}")  # 打印 x_image 的形状
        # print(f"x_numeric shape before fc: {x_numeric.shape}")  # 打印 x_numeric 的形状

        # 将 Transformer 输出特征调整到与 ResNet 输出相同的维度
        x_numeric = self.fc_transformer(x_numeric)

        # 使用池化层将 x_numeric 的时间维度缩减
        x_numeric = x_numeric.mean(dim=0).unsqueeze(0).expand(x_image.size(0), -1)

        # 再次检查形状
        # print(f"x_numeric shape after pooling: {x_numeric.shape}")

        # 拼接两个特征
        x_combined = torch.cat((x_image, x_numeric), dim=1)
        x_combined = nn.ReLU()(self.fc1(x_combined))
        x_out = self.fc2(x_combined)
        return x_out

class MultimodalTransformerMobileNet(nn.Module):
    def __init__(self, num_numeric_features, embed_dim, num_heads, ff_dim, num_layers, num_classes=2):
        super(MultimodalTransformerMobileNet, self).__init__()
        self.transformer = TimeSeriesInParformer(input_dim=1, embed_dim=embed_dim, num_heads=num_heads,
                                                 ff_dim=ff_dim, num_layers=num_layers)
        self.mobilenet = mobilenet_v2(pretrained=True)  # 使用 MobileNetV2
        num_ftrs = self.mobilenet.classifier[1].in_features  # 获取 MobileNetV2 全连接层的输入特征数
        self.mobilenet.classifier = nn.Identity()  # 移除最后的全连接层

        # 新增全连接层用于对齐 Transformer 输出的特征维度
        self.fc_transformer = nn.Linear(embed_dim, num_ftrs)  # 将 Transformer 输出调整为 MobileNet 输出的大小

        self.fc1 = nn.Linear(num_ftrs + num_ftrs, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x_image, x_numeric):
        x_image = self.mobilenet(x_image)
        x_numeric = self.transformer(x_numeric).mean(dim=0)  # 对 Transformer 的输出进行池化

        # 将 Transformer 输出特征调整到与 MobileNet 输出相同的维度
        x_numeric = self.fc_transformer(x_numeric)

        # 使用池化层将 x_numeric 的时间维度缩减
        x_numeric = x_numeric.mean(dim=0).unsqueeze(0).expand(x_image.size(0), -1)

        # 拼接两个特征
        x_combined = torch.cat((x_image, x_numeric), dim=1)
        x_combined = nn.ReLU()(self.fc1(x_combined))
        x_out = self.fc2(x_combined)
        return x_out

# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载数值数据
NumericData_dir = r'/home/hr/pro/multi_drug/transformer/data/mlxy2000_train/test_269/csv'
numeric_data, labels, filenames = load_NumericData(NumericData_dir)

# 初始化多模态数据集
dataset = MultimodalDataset(
    image_dir=r'/home/hr/pro/multi_drug/transformer/data/mlxy2000_train/test_269/image',
    numeric_data=numeric_data,
    labels=labels,
    transform=transform
)
test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# 实例化模型
embed_dim = 64
num_heads = 4
ff_dim = 128
num_layers = 2
model = MultimodalTransformerMobileNet(num_numeric_features=150, embed_dim=embed_dim, num_heads=num_heads,
                                    ff_dim=ff_dim, num_layers=num_layers)

# 加载保存的权重
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# 设置自定义的分类阈值
threshold = 0.8189

# 在测试代码部分前添加：保存所有文件名
all_filenames = []

# 测试模型并计算分类指标
correct = 0
total = 0
all_preds = []
all_labels = []
all_probs = []

for idx, (images, numerics, labels) in enumerate(test_loader):
    images, numerics, labels = images.to(device), numerics.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(images, numerics)
        probs = torch.softmax(outputs, dim=1)  # 计算类别的概率
        predicted = (probs[:, 1] >= threshold).long()  # 使用自定义阈值进行分类
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        # 记录文件名
        all_filenames.extend([dataset.image_files[i + idx * test_loader.batch_size] for i in range(len(labels))])

# 输出使用的分类阈值
print(f'Classification Threshold: {threshold}')

# 计算评估指标
test_accuracy = 100 * correct / total
f1 = f1_score(all_labels, all_preds, average='weighted')
balanced_acc = balanced_accuracy_score(all_labels, all_preds)
kappa = cohen_kappa_score(all_labels, all_preds)

print(f'Test Accuracy: {test_accuracy}%')
print(f'F1 Score: {f1}')
print(f'Balanced Accuracy: {balanced_acc}')
print(f'Cohen\'s Kappa: {kappa}')
print('Classification Report:')
print(classification_report(all_labels, all_preds, zero_division=0))

# 输出文件名、真实标签和预测标签
print('\nSample Filename - True Label - Predicted Label')
for filename, true_label, pred_label in zip(all_filenames, all_labels, all_preds):
    print(f'{filename} - {true_label} - {pred_label}')
import pandas as pd

# 创建一个 DataFrame 保存文件名、真实标签和预测标签
results_df = pd.DataFrame({
    'Filename': all_filenames,
    'True Label': all_labels,
    'Predicted Label': all_preds
})

# 将 DataFrame 保存为 CSV 文件
results_df.to_csv('classification_results.csv', index=False, encoding='utf-8')

print('Results have been saved to classification_results.csv')
