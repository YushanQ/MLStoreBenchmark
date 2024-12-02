from imagenetv2_pytorch import ImageNetV2Dataset
from torch.utils.data import DataLoader
import torch
import torch.cuda.nvtx as nvtx
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights
from PIL import Image
import numpy as np
import os


# 定义自定义数据集类
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 转换为三通道
    transforms.Resize(256),  # 调整大小以符合ResNet50的输入
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class ImageNetV2Dataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root=root, transform=transform)
        self.paths = []  # Store image paths

        # Recursively collect all image paths from subdirectories
        for label in sorted(os.listdir(root)):  # Assuming subfolders are named 0, 1, ..., 999
            label_dir = os.path.join(root, label)
            if os.path.isdir(label_dir):
                for file_name in sorted(os.listdir(label_dir)):
                    file_path = os.path.join(label_dir, file_name)
                    if os.path.isfile(file_path):
                        self.paths.append(file_path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        label = int(os.path.basename(os.path.dirname(img_path)))
        return self.transform(img), label


dataset = ImageNetV2Dataset(root="ImageNetV2-matched-frequency", transform=transform) # supports matched-frequency, threshold-0.7, top-images variants
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4) # use whatever batch size you wish


for i, (inputs, labels) in enumerate(dataloader):
    print(f"iteration {i} ")
    if i > 20:
        break
