from torchvision.transforms.functional import to_tensor, to_pil_image
import torch
import torch.nn.functional as F

def enhance_high_frequency(image, weight=0.5):
    """
    提取图像的高频细节部分并加权叠加回原图，增强图像清晰度。
    :param image: 输入图像 (PIL Image)
    :param weight: 高频增强的权重
    :return: 增强后的图像 (PIL Image)
    """
    # 转换为 Tensor，形状为 [C, H, W]
    image_tensor = to_tensor(image)
    
    # 定义 Sobel 滤波器
    sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).unsqueeze(0)
    sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32).unsqueeze(0)
    
    # 适配多通道 (RGB)，对每个通道分别应用 Sobel 滤波器
    sobel_x = sobel_x.expand(image_tensor.shape[0], 1, 3, 3)  # [C, 1, 3, 3]
    sobel_y = sobel_y.expand(image_tensor.shape[0], 1, 3, 3)  # [C, 1, 3, 3]
    
    # 增加批次维度 [1, C, H, W]
    image_tensor = image_tensor.unsqueeze(0)
    
    # 计算高频部分 (边缘)
    edge_x = F.conv2d(image_tensor, weight=sobel_x, padding=1, groups=image_tensor.shape[1])
    edge_y = F.conv2d(image_tensor, weight=sobel_y, padding=1, groups=image_tensor.shape[1])
    high_freq = torch.sqrt(edge_x**2 + edge_y**2).squeeze(0)  # 移除批次维度 [C, H, W]
    
    # 增强高频细节部分
    enhanced_tensor = torch.clamp(image_tensor.squeeze(0) + weight * high_freq, 0, 1)
    
    # 转换回 PIL Image
    enhanced_image = to_pil_image(enhanced_tensor)
    return enhanced_image
