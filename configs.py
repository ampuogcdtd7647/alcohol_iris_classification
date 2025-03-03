import argparse
import yaml
import os
import numpy as np
import pywt

class Config:
    def __init__(self):
        # 数据路径相关
        self.train_image_dir = r'/NEU/hr/pro/drug/paper_drug/transformer/paper_data/train_image'
        self.train_numeric_data_dir = r'/NEU/hr/pro/drug/paper_drug/transformer/paper_data/train_csv'
        self.output_dir = r'/NEU/hr/pro/drug/paper_drug/transformer/result_image'
        self.output_file = 'testsmall_predictions_vs_labels.csv'
        
        # 数据处理相关
        self.image_size = (224, 224)
        self.batch_size = 32
        self.augment_probability = 0.5
        self.rotation_degrees = 30
        self.gaussian_sigma = 2
        self.cutoff_frequency = 0.1
        self.max_numeric_length = 128

        # 模型参数
        self.embed_dim = 128
        self.num_heads = 8
        self.ff_dim = 128
        self.num_layers = 2
        self.num_classes = 2
        self.pretrained_resnet = True
        self.pretrained_mobilenet = True
        self.input_nc = 3
        self.dwt_out_nc = 64
        self.ffc_out_nc = 128
        self.wavelet = 'morl'  # 使用 Morlet 小波
        self.scales = np.arange(1, 31)  # 定义 CWT 的尺度范围
        
        # 训练相关
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.patience = 8
        self.initial_alpha = 1.0
        self.target_fpr = 0.0001
        self.train_val_test_split = (0.8, 0.2)
        self.random_seed = 77

        #测试相关
        self.test_numeric_data_dir=r'/NEU/hr/pro/drug/paper_drug/transformer/paper_data/test_csv'
        self.test_image_dir=r'/NEU/hr/pro/drug/paper_drug/transformer/paper_data/test_image'
        
        # 阈值调优
        self.optimal_fpr = 0.01

    def get_output_path(self):
        return os.path.join(self.output_dir, self.output_file)
