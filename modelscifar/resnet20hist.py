

'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable
import sys
#sys.path.append('../')
import quantizer as Q

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


def conv(input, output, kernel_size=3, stride=1, padding=1, bias=False,nbits=2):
    return Q.Conv2dLSQ_sym(input, output, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, nbits=nbits)




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
        self.conv1 = conv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
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
        out, xqe1 = self.conv1(x)
        out = F.relu(self.bn1(out))
        out, xqe2 = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        hist1 = torch.histc(xqe1.flatten(), bins=4, min=0, max=3)
        hist2 = torch.histc(xqe2.flatten(), bins=4, min=0, max=3)
        return out, hist1+hist2


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = BasicBlock(self.in_planes, 16, stride=1)
        self.in_planes = 16 * block.expansion
        self.layer2 = BasicBlock(self.in_planes, 16, stride=1)
        self.in_planes = 16 * block.expansion
        self.layer3 = BasicBlock(self.in_planes, 16, stride=1)
        self.in_planes = 16 * block.expansion


        self.layer4 = BasicBlock(self.in_planes, 32, stride=2)
        self.in_planes = 32 * block.expansion
        self.layer5 = BasicBlock(self.in_planes, 32, stride=1)
        self.in_planes = 32 * block.expansion
        self.layer6 = BasicBlock(self.in_planes, 32, stride=1)
        self.in_planes = 32 * block.expansion


        self.layer7 = BasicBlock(self.in_planes, 64, stride=2)
        self.in_planes = 64 * block.expansion
        self.layer8 = BasicBlock(self.in_planes, 64, stride=1)
        self.in_planes = 64 * block.expansion
        self.layer9 = BasicBlock(self.in_planes, 64, stride=1)
        self.in_planes = 64 * block.expansion

        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out,hist1 = self.layer1(out)
        out,hist2 = self.layer2(out)
        out,hist3 = self.layer3(out)
        out,hist4 = self.layer4(out)
        out,hist5 = self.layer5(out)
        out,hist6 = self.layer6(out)
        out,hist7 = self.layer7(out)
        out,hist8 = self.layer8(out)
        out,hist9 = self.layer9(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        hist = hist1+hist2+hist3+hist4+hist5+hist6+hist7+hist8+hist9
        #print(hist1+hist2+hist3)
        return out#,hist


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()


