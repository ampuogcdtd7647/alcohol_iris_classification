import torch
import torch.nn as nn
import torch.nn.functional as F

# DWT：离散小波变换，提取低频和高频特征
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        # 假设输入尺寸为 [B, C, H, W]
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4  # 低频
        x_HL = -x1 - x2 + x3 + x4  # 高频
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4
        return x_LL, torch.cat((x_HL, x_LH, x_HH), 1)

# 卷积和降维：对低频和高频特征进行卷积降维
class DWT_transform(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt = DWT()
        self.conv1x1_low = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv1x1_high = nn.Conv2d(in_channels * 3, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        dwt_low_frequency, dwt_high_frequency = self.dwt(x)
        dwt_low_frequency = self.conv1x1_low(dwt_low_frequency)  # 低频特征降维
        dwt_high_frequency = self.conv1x1_high(dwt_high_frequency)  # 高频特征降维
        return dwt_low_frequency, dwt_high_frequency

# Fast Fourier Convolution (FFC)
class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, fft_norm='ortho'):
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2, kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]
        
        # 进行FFT转换
        ffted = torch.fft.rfftn(x, dim=(-2, -1), norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))
        
        # 逆傅里叶变换
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        output = torch.fft.irfftn(ffted, s=x.shape[-2:], dim=(-2, -1), norm=self.fft_norm)
        return output

# 频谱变换：通过频谱变换进一步提取特征
class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpectralTransform, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.fu = FourierUnit(out_channels // 2, out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.fu(x)
        x = self.conv2(x)
        return x

# 特征融合
class FeatureExtractor(nn.Module):
    def __init__(self, input_nc, dwt_out_nc, ffc_out_nc):
        super().__init__()
        self.dwt_transform = DWT_transform(input_nc, dwt_out_nc)
        self.ffc_resnet = FourierUnit(dwt_out_nc, ffc_out_nc)

    def forward(self, x):
        # 使用 DWT 提取低频和高频特征
        dwt_low, dwt_high = self.dwt_transform(x)

        ffc_features = self.ffc_resnet(dwt_low)

        ffc_features_high = self.ffc_resnet(dwt_high)

        # 融合低频（FFC）和高频（DWT）特征
        combined_features = torch.cat((ffc_features, ffc_features_high), dim=1)

        return combined_features

# 主函数
def main():
    # 输入数据
    input_image = torch.randn(32, 3, 224, 224)  # 输入为 [batch_size, channels, height, width]

    # 初始化特征提取器
    feature_extractor = FeatureExtractor(input_nc=3, dwt_out_nc=64, ffc_out_nc=96)

    # 提取图像特征
    output_features = feature_extractor(input_image)

    # 输出特征维度
    print("输出特征维度:", output_features.shape)  # 输出维度为 (32, 192, 112, 112)

if __name__ == "__main__":
    main()
