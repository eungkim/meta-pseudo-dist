import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torch.nn.init as init


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, input_channels=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv0 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.fc1 = nn.Linear(512 * block.expansion, 256)
        self.bn1 = nn.BatchNorm1d(256)
        # self.fc2 = nn.Linear(1024, 1024)
        # self.bn2 = nn.BatchNorm1d(1024)

        self.fc_mu = nn.Linear(256, 256)
        # self.fc_logvar = nn.Linear(1024, 1024)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # encoder
        x = self.conv0(x)
        x = self.bn0(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, kernel_size=7, stride=2)
        x = x.view(x.size(0), -1)
        
        # mu, log_var
        mu = F.relu(self.bn1(self.fc1(x)))
        # x = F.relu(self.bn2(self.fc2(x)))
        mu = self.fc_mu(mu)
        # log_var = self.fc_logvar(x)

        # return mu, log_var
        return mu, x


def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


class Teacher(nn.Module):
    def __init__(self):
        super(Teacher, self).__init__()
        expansion = 4
        self.fc1 = nn.Linear(512 * expansion, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)

        return x


class Decoder(nn.Module):
    def __init__(self, block=Bottleneck, num_blocks=[5, 5, 5]):
        super(Decoder, self).__init__()
        # decoder
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        
        # reparametrize
        eps = Variable(std.data.new(std.size()).normal())
        x = eps.mul(std).add_(mu)

        # to edit
        # decode
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = F.avg_pool2d(out, out.size()[3])
        # out = out.view(out.size(0), -1)
        # out = F.relu(self.fc1_bn(self.fc1_enc(out)))
        # out = F.relu(self.fc2_bn(self.fc2_enc(out)))
        # mu = self.fc_mu(out)
        # logvar = self.fc_logvar(out)
        # std = logvar.mul(0.5).exp_()

        return x


class Student(nn.Module):
    def __init__(self, encoder, decoder):
        super(Student, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        mu, log_var = self.encoder(x)
        x_hat = self.decoder(mu, log_var)
        # to edit
        return x_hat, mu, log_var