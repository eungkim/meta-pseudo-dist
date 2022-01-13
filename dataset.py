import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
import numpy as np


def build_dataset(batch_size=256, num_worker=4, path="data/ImageNet/"):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
    ])
    valid_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
    ])
    
    train_dataset = BiAugImageNet(path, split='train', transform=train_transform)
    train_acc_dataset = AugImageNet(path, split='train', transform=train_transform)
    valid_dataset = datasets.ImageNet(path, split='val', transform=valid_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_worker)
    train_meta_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_worker)
    train_acc_loader = DataLoader(train_acc_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_worker)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_worker)

    return train_loader, train_meta_loader, train_acc_loader, valid_loader 


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
aug_transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        normalize,
])
class BiAugImageNet(datasets.ImageNet):
    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)
        img = self.transform(sample)
        img1 = aug_transform(img)
        img2 = aug_transform(img)

        return img1, img2
        
class AugImageNet(datasets.ImageNet):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        img = self.transform(sample)
        img = aug_transform(img)

        return img, target