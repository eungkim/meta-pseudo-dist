import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

"""
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

Thanks to the author, Yerlan Idelbayev
"""

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, latent):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        if latent<=64:
            self.out_dim = 64
        elif latent<=256:
            self.out_dim = 256
        elif latent<=1024:
            self.out_dim = 1024
        else:
            self.out_dim = 4096

        self.fc1 = nn.Linear(self.out_dim, latent)
        self.fc1_bn = nn.BatchNorm1d(latent)
        self.fc2 = nn.Linear(latent, latent)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # [B, 64, 8, 8]
        if self.out_dim==64:
            x = F.avg_pool2d(x, x.size()[3]) #[B, 64, 1, 1]
        elif self.out_dim==256:
            x = F.adaptive_avg_pool2d(x, (2, 2)) #[B, 64, 2, 2] to [B, 256]
        elif self.out_dim==1024:
            x = F.adaptive_avg_pool2d(x, (4, 4)) #[B, 64, 4, 4] to [B, 1024]
        else:
            pass

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc2(x)
        return x


def resnet32(latent):
    return ResNet(BasicBlock, [5, 5, 5], latent)

class DoubleResNet(nn.Module):
    def __init__(self, model1, model2):
        super(DoubleResNet, self).__init__()
        self.model1 = model1
        self.model2 = model2
    
    def forward(self, x1, x2):
        x1 = self.model1(x1)
        x2 = self.model2(x2)
        return x1, x2

def dresnet32(latent):
    return DoubleResNet(ResNet(BasicBlock, [5, 5, 5], latent), ResNet(BasicBlock, [5, 5, 5], latent))

class TResNet(nn.Module):
    def __init__(self, block, num_blocks, latent):
        super(TResNet, self).__init__()
        self.in_planes = 16
        self.latent = latent

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.fc1 = nn.Linear(64*8*8, 64*8*8)
        self.fc1_bn = nn.BatchNorm1d(64*8*8)
        self.fc2 = nn.Linear(64*8*8, 64*8*8)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # [B, 64*8*8]
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc2(x)
        x = x.view(x.size(0), self.latent, -1)
        return x


def t_resnet32(latent):
    return TResNet(BasicBlock, [5, 5, 5], latent)
