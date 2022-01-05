import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
import numpy as np


def build_dataset(name="ImageNet", download=False, path="data/ImageNet/"):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    valid_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    
    train_dataset = datasets.ImageNet(path, split='train', download=download, transform=train_transform)
    test_dataset = datasets.ImageNet(path, split='test', download=download, transform=valid_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    train_meta_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, train_meta_loader, test_loader