import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math
from .bwn import weight_norm as wn
from .trelu import TReLU
__all__ = ['resnet_wn_trelu']

p = 2


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return wn(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=1, bias=True), p=p)


def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = TReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = wn(nn.Conv2d(inplanes, planes,
                                  kernel_size=1, bias=True), p=p)
        self.conv2 = wn(nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                  padding=1, bias=True), p=p)
        self.conv3 = wn(nn.Conv2d(planes, planes * 4,
                                  kernel_size=1, bias=True), p=p)
        self.relu = TReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_WN(nn.Module):

    def __init__(self):
        super(ResNet_WN, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                wn(nn.Conv2d(self.inplanes, planes * block.expansion,
                             kernel_size=1, stride=stride, bias=True), p=p),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet_WN_imagenet(ResNet_WN):

    def __init__(self, num_classes=1000,
                 block=Bottleneck, layers=[3, 4, 23, 3]):
        super(ResNet_WN_imagenet, self).__init__()
        self.inplanes = 64
        self.conv1 = wn(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                  bias=True), p=p)
        self.relu = TReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        init_model(self)
        scale = 2
        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': scale * 1e-1,
             'weight_decay': 0, 'momentum': 0.9},
            {'epoch': 30, 'lr': scale * 1e-2},
            {'epoch': 60, 'lr': scale * 1e-3},
            {'epoch': 90, 'lr': scale * 1e-4}
        ]


class ResNet_WN_cifar10(ResNet_WN):

    def __init__(self, num_classes=10,
                 block=BasicBlock, depth=18):
        super(ResNet_WN_cifar10, self).__init__()
        self.inplanes = 16
        n = int((depth - 2) / 6)
        self.conv1 = wn(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                                  bias=True), p=p)
        self.relu = TReLU(inplace=True)
        self.maxpool = lambda x: x
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.scale = nn.Parameter(torch.Tensor([10]))
        self.fc = nn.Linear(64, num_classes)  # , p = p)

        init_model(self)
        scale = 1
        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': scale * 1e-1,
             'weight_decay': 0, 'momentum': 0.9},
            {'epoch': 81, 'lr': scale * 1e-2},
            {'epoch': 122, 'lr': scale * 1e-3, 'weight_decay': 0},
            {'epoch': 164, 'lr': scale * 1e-4}
        ]


def resnet_wn_trelu(**kwargs):
    num_classes, depth, dataset = map(
        kwargs.get, ['num_classes', 'depth', 'dataset'])
    if dataset == 'imagenet':
        num_classes = num_classes or 1000
        depth = depth or 50
        if depth == 18:
            return ResNet_WN_imagenet(num_classes=num_classes,
                                      block=BasicBlock, layers=[2, 2, 2, 2])
        if depth == 34:
            return ResNet_WN_imagenet(num_classes=num_classes,
                                      block=BasicBlock, layers=[3, 4, 6, 3])
        if depth == 50:
            return ResNet_WN_imagenet(num_classes=num_classes,
                                      block=Bottleneck, layers=[3, 4, 6, 3])
        if depth == 101:
            return ResNet_WN_imagenet(num_classes=num_classes,
                                      block=Bottleneck, layers=[3, 4, 23, 3])
        if depth == 152:
            return ResNet_WN_imagenet(num_classes=num_classes,
                                      block=Bottleneck, layers=[3, 8, 36, 3])

    elif dataset == 'cifar10':
        num_classes = num_classes or 10
        depth = depth or 56
        return ResNet_WN_cifar10(num_classes=num_classes,
                                 block=BasicBlock, depth=depth)
    elif dataset == 'cifar100':
        num_classes = num_classes or 100
        depth = depth or 56
        return ResNet_WN_cifar10(num_classes=num_classes,
                                 block=BasicBlock, depth=depth)