import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from pretrainedmodels import resnet50
from torchvision.models import resnet18, mobilenet_v2
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score, classification_report
from torchvision.transforms import transforms
import os
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score
from torch.utils.data import random_split, DataLoader
from scipy.ndimage import gaussian_filter1d
import random
from MyFunctions import enhance_high_frequency
from MyModule.DWT_FFC import DWT, DWT_transform, FeatureExtractor
from MyModule.MLP import SimpleMLP
from MyModule.Attention import Attention 
import configs
import torch
import os
import pywt

import re

# 加载配置
config = configs.Config()

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# 设置Python hash 随机化种子，确保字符串操作的一致性
os.environ['PYTHONHASHSEED'] = '42'

# 设置Python标准库中的随机数种子
random.seed(42)

# 设置NumPy随机数生成器的种子
np.random.seed(42)

# 设置PyTorch随机数生成器的种子
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # if you are using multi-GPU

# 确保PyTorch的确定性行为
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


    
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
            column_name = 'quotient_rx'

            # 检查列是否存在
            if column_name in df.columns:
                signal = df[column_name].copy()

                # 计算均值和方差
                mean_signal = signal.mean()
                std_signal = signal.std()

                # 将均值和方差作为输出特征
                features = [mean_signal, std_signal]

                # 将处理后的特征添加到数据集中
                data.append(features)
            
            # 标签分类：'normal' 作为标签0，其他为标签1
            label = 0 if 'normal' in filename else 1
            labels.append(label)
            filenames.append(filename)

    return data, labels, filenames

# 自定义数据集类
class MultimodalDataset(Dataset):
    def __init__(self, image_dir, numeric_data, labels, transform=None, augment_transform=None, enhance_high_freq = False):
        self.image_dir = image_dir
        self.transform = transform
        self.augment_transform = augment_transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))], key=sort_key)
        self.numeric_data = numeric_data
        self.labels = labels
        self.enhance_high_freq = enhance_high_freq  

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        # 应用高频增强
        if self.enhance_high_freq:
            image = enhance_high_frequency(image)
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


class AdaptiveFPRFNRControlledLoss(nn.Module):
    def __init__(self, base_loss, target_fpr=0.01, target_fnr=0.01, initial_alpha=1.0, initial_beta=1.0, learnable_alpha=False, learnable_beta=False):
        """
        自适应调整的自定义损失函数，用于控制类别 0 的误报率（FPR）和类别 1 的误报率（FNR）。

        :param base_loss: 基础损失函数，如交叉熵损失 (nn.CrossEntropyLoss)
        :param target_fpr: 目标误报率，即类别 0 的误报率水平
        :param target_fnr: 目标漏报率，即类别 1 的漏报率水平
        :param initial_alpha: 控制 FPR 的初始 alpha 值
        :param initial_beta: 控制 FNR 的初始 beta 值
        :param learnable_alpha: 是否将 alpha 设置为可学习参数
        :param learnable_beta: 是否将 beta 设置为可学习参数
        """
        super(AdaptiveFPRFNRControlledLoss, self).__init__()
        self.base_loss = base_loss
        self.target_fpr = target_fpr
        self.target_fnr = target_fnr

        # 设置 alpha 为控制 FPR 的参数
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(initial_alpha, dtype=torch.float32))
        else:
            self.alpha = initial_alpha

        # 设置 beta 为控制 FNR 的参数
        if learnable_beta:
            self.beta = nn.Parameter(torch.tensor(initial_beta, dtype=torch.float32))
        else:
            self.beta = initial_beta

    def forward(self, outputs, labels):
        # 计算基础损失
        base_loss = self.base_loss(outputs, labels)

        # 使用 softmax 将输出转换为概率
        probs = torch.softmax(outputs, dim=1)

        # 预测标签
        preds = probs[:, 1] >= 0.5

        # 计算 FPR（类别 0 的误报率）
        fp = ((preds == 1) & (labels == 0)).float().sum()  # 假阳性数量
        tn = ((preds == 0) & (labels == 0)).float().sum()  # 真阴性数量
        current_fpr = fp / (fp + tn + 1e-8)  # 防止除以零

        # 计算 FNR（类别 1 的漏报率）
        fn = ((preds == 0) & (labels == 1)).float().sum()  # 假阴性数量
        tp = ((preds == 1) & (labels == 1)).float().sum()  # 真阳性数量
        current_fnr = fn / (fn + tp + 1e-8)  # 防止除以零

        # 如果当前 FPR 大于目标 FPR，加入 FPR 惩罚项
        fpr_penalty = self.alpha * torch.clamp(current_fpr - self.target_fpr, min=0)

        # 如果当前 FNR 大于目标 FNR，加入 FNR 惩罚项
        fnr_penalty = self.beta * torch.clamp(current_fnr - self.target_fnr, min=0)

        # 总损失 = 基础损失 + FPR 惩罚 + FNR 惩罚
        total_loss = base_loss + fpr_penalty + fnr_penalty

        return total_loss



# 定义多模态模型，集成 Transformer 和 MobileNet
import torch.nn.functional as F

class MultimodalTransformerMobileNet(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, wavelet='morl', scales=np.arange(1, 31), num_classes=2):
        super(MultimodalTransformerMobileNet, self).__init__()
        self.feature_extractor = FeatureExtractor(input_nc=3, dwt_out_nc=64, ffc_out_nc=96)
        self.mlp = SimpleMLP(input_dim=2, hidden_dim=64)
        # 初始化自注意力层
        self.attention_layer = Attention(embed_dim = 257)
        # 初始化 MobileNet 的 features 部分
        mobilenet = mobilenet_v2(pretrained=True)
        self.mobilenet_features = mobilenet.features
        self.mobilenet_pool = nn.AdaptiveAvgPool2d((112, 112))  # 调整到与 FeatureExtractor 输出匹配
        self.conv_adjust = nn.Conv2d(1280, 192, kernel_size=1, stride=1, padding=0)  # 调整通道数为192

        # 归一化层
        self.image_norm = nn.LayerNorm([192, 7, 7])  # 对图像特征进行归一化
        self.numeric_norm = nn.LayerNorm(1)  # 对时间序列特征进行归一化

        # 输出调整
        self.fc_transformer = nn.Linear(embed_dim, 256)  # 将 Transformer 输出调整为与图像特征匹配
        self.fc3 = nn.Linear(256, 1)  # 映射到单个数值
        self.pooling_layer = nn.AdaptiveAvgPool2d((7, 7))

        # 全连接层，用于分类
        self.image_mlp = nn.Sequential(
            nn.Linear(192 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)  # 输出类别概率
        )

        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x_image, x_numeric):
        # print(x_numeric.shape) # [32, 2, 1]
        # --- 图像特征提取 ---
        batch_size = x_image.shape[0]
        extracted_features = self.feature_extractor(x_image)  # 输出 [batch_size, 192, 112, 112]
        extracted_features = self.pooling_layer(extracted_features)  # [batch_size, 192, 7, 7]
        mobilenet_features = self.mobilenet_features(x_image)  # 输出 [batch_size, 1280, 7, 7]
        mobilenet_features = self.conv_adjust(mobilenet_features)  # 调整通道数为 [batch_size, 192, 7, 7]
        
        # Shortcut 连接
        shortcut_features = extracted_features + mobilenet_features  # [batch_size, 192, 7, 7]
        shortcut_features = self.image_norm(shortcut_features)  # 归一化

        # --- 时间序列特征提取 ---
        x_numeric = x_numeric.squeeze(-1)
        x_numeric = self.mlp(x_numeric)  # [batch_size, class_number]
        
        # --- 分类 ---
        x_image_flattened = shortcut_features.view(batch_size, -1)
        logits_image = self.image_mlp(x_image_flattened)  # [batch_size, num_classes]
        # 加权平均融合（最终 logits 进行 softmax
        fused_logits = self.fusion_weight * logits_image + (1 - self.fusion_weight) * x_numeric # [batch_size, num_classes]

        # 转换为概率
        probabilities = torch.softmax(fused_logits, dim=-1)

        return probabilities




# 模型初始化
model = MultimodalTransformerMobileNet( 
    embed_dim=config.embed_dim, 
    num_heads=config.num_heads, 
    ff_dim=config.ff_dim, 
    num_layers=config.num_layers, 
    wavelet=config.wavelet,  # 传递 wavelet 参数
    scales=config.scales,    # 传递 scales 参数
    num_classes=config.num_classes
)


# 检查模型的参数数据类型
for param in model.parameters():
    print(param.dtype)
    break  # 打印一次即可，因为所有参数通常都是同一类型

# 确保模型运行在 GPU 上
if torch.cuda.is_available():
    model = model.cuda()
    print("使用 GPU 进行训练")
else:
    print("使用 CPU 进行训练")

# 损失函数和优化器
criterion = AdaptiveFPRFNRControlledLoss(
    base_loss=nn.CrossEntropyLoss(),
    target_fpr=config.target_fpr,
    initial_alpha=config.initial_alpha,
    learnable_alpha=False
)
optimizer = optim.Adam(
    list(model.parameters()) + list(criterion.parameters()), 
    lr=config.learning_rate, 
    weight_decay=config.weight_decay
)

# 图像预处理和数据增强
preprocess_transforms = transforms.Compose([
    transforms.Resize(config.image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=config.augment_probability),
    transforms.RandomRotation(degrees=config.rotation_degrees),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
])



# 数据集加载
train_numeric_data, train_labels, train_filenames = load_NumericData(
    data_dir=config.train_numeric_data_dir, # 在配置文件configs.py中修改路径
    max_length=config.max_numeric_length, 
    cutoff=config.cutoff_frequency
)
full_train_dataset = MultimodalDataset(
    image_dir=config.train_image_dir,
    numeric_data=train_numeric_data,
    labels=train_labels,
    transform=preprocess_transforms,
    # augment_transform=augmentation_transforms,
    enhance_high_freq=True
)
test_numeric_data, test_labels, test_filenames = load_NumericData(
    data_dir=config.test_numeric_data_dir, # 在配置文件configs.py中修改路径
    max_length=config.max_numeric_length, 
    cutoff=config.cutoff_frequency
)

full_test_dataset = MultimodalDataset(
    image_dir=config.test_image_dir,
    numeric_data=train_numeric_data,
    labels=test_labels,
    transform=preprocess_transforms,
    # augment_transform=augmentation_transforms,
    enhance_high_freq=True
)

total_size = len(full_train_dataset)

# 数据集划分
train_size = int(config.train_val_test_split[0] * len(full_train_dataset))
val_size = int(config.train_val_test_split[1] * len(full_train_dataset))
generator = torch.Generator().manual_seed(config.random_seed)
train_dataset, val_dataset = random_split(
    full_train_dataset, [train_size, val_size], generator=generator
)


test_filenames = full_test_dataset.image_files

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(full_test_dataset, batch_size=32, shuffle=False)

# Training and validation loop
patience = 8
best_loss = float('inf')
best_epoch = 0
stop = False
epoch = 0

while not stop:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, numerics, labels in train_loader:
        if torch.cuda.is_available():
            images, numerics, labels = images.cuda(), numerics.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(images, numerics)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    print(f'Epoch {epoch}, Training Loss: {avg_loss}, Training Accuracy: {train_accuracy}%')

    # Validation
    model.eval()
    val_loss = 0.0
    for images, numerics, labels in val_loader:
        if torch.cuda.is_available():
            images, numerics, labels = images.cuda(), numerics.cuda(), labels.cuda()
        with torch.no_grad():
            outputs = model(images, numerics)
            loss = criterion(outputs, labels)
        val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f'Epoch {epoch}, Validation Loss: {avg_val_loss}')

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), '/NEU/hr/pro/drug/paper_drug/transformer/result_image/best_model_wavelets.pth')
    elif epoch - best_epoch >= patience:
        print("Stopping early due to no improvement")
        stop = True

    epoch += 1

# Load the best model and evaluate on the test set
model.load_state_dict(torch.load('/NEU/hr/pro/drug/paper_drug/transformer/result_image/best_model_wavelets.pth'))
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []
all_probs = []
for images, numerics, labels in test_loader:
    if torch.cuda.is_available():
        images, numerics, labels = images.cuda(), numerics.cuda(), labels.cuda()
    with torch.no_grad():
        outputs = model(images, numerics)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

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

# Convert all_probs to NumPy array
all_probs = np.array(all_probs)

# Calculate FPR and TPR for label 0 (normal)
fpr, tpr, thresholds = roc_curve(all_labels, all_probs[:, 1])

# Find the optimal threshold where FPR < 0.01
target_fpr = 0.01
optimal_idx = np.where(fpr <= target_fpr)[0][-1]
optimal_threshold = thresholds[optimal_idx]

# Classify based on the optimal threshold
optimal_predictions = (all_probs[:, 1] >= optimal_threshold).astype(int)

# Calculate accuracy and other metrics
accuracy = accuracy_score(all_labels, optimal_predictions)
precision = precision_score(all_labels, optimal_predictions, pos_label=1)
recall = recall_score(all_labels, optimal_predictions, pos_label=1)
recall_normal = recall_score(all_labels, optimal_predictions, pos_label=0)
f1 = f1_score(all_labels, optimal_predictions, pos_label=1)

# Print the results
print(f'Optimal Threshold: {optimal_threshold}')
print(f'Accuracy at optimal threshold: {accuracy}')
print(f'Precision at optimal threshold: {precision}')
print(f'Recall at optimal threshold: {recall}')
print(f'Recall of normal at optimal threshold: {recall_normal}')
print(f'F1 Score at optimal threshold: {f1}')

import csv

# 指定保存的文件夹路径和文件名
output_dir = r'/NEU/hr/pro/drug/paper_drug/transformer/result_image'  # 替换为你的目标文件夹路径
output_file = os.path.join(output_dir, 'testsmall_predictions_vs_labels.csv')

# 如果文件夹不存在，则创建文件夹
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Directory {output_dir} created.")

# 保存测试集真实标签和预测标签到 CSV 文件
with open(output_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Filename', 'True Label', 'Predicted Label'])
    for i, filename in enumerate(test_filenames):
        csvwriter.writerow([filename, all_labels[i], all_preds[i]])

print(f'Test set filenames, true labels, and predicted labels have been saved to {output_file}')

