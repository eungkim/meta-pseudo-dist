import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
import numpy as np
from PIL import Image


def build_dataset(type_dataset="cifar10", batch_size=256, num_worker=4, path="data/ImageNet/"):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    if type_dataset=="cifar10": ###################to edit
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        train_dataset = BiAugCIFAR10(root=path, train=True, transform=train_transform, download=True)
        train_acc_dataset = AugCIFAR10(root=path, train=True, transform=train_transform, download=True)
        valid_dataset = datasets.CIFAR10(root=path, train=False, transform=valid_transform, download=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_worker)
        train_meta_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_worker)
        train_acc_loader = DataLoader(train_acc_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_worker)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_worker)

    elif type_dataset=="imagenet":
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

    else:
        print("wrong dataset! chosse imagenet of cifar10")

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

class BiAugCIFAR10(datasets.CIFAR10):
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        img1 = self.transform(img)
        img2 = self.transform(img)

        return img1, img2
        
class AugCIFAR10(datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        img = self.transform(img)

        return img, target