import torch
from torch import nn
from torch.nn import functional as F
# from MobileNetV1 import Flatten


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class InvertedResidualLinearBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride):
        super(InvertedResidualLinearBottleneck, self).__init__()
        self.use_res = in_channels == out_channels and stride == 1
        self.conv1 = nn.Conv2d(in_channels, in_channels*expansion, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels*expansion)
        self.conv2 = nn.Conv2d(in_channels*expansion, in_channels*expansion,
                               kernel_size=3, stride=stride, padding=1,
                               groups=in_channels*expansion)
        self.bn2 = nn.BatchNorm2d(in_channels*expansion)
        self.conv3 = nn.Conv2d(in_channels*expansion, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        x = F.relu6(self.bn1(self.conv1(x)))
        x = F.relu6(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.use_res is True:
            x += identity

        return x


ARCH = [
    [None, 32, 1, 2],
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
    [None, 1280, 1, 1]
]


class MobileNetV2(nn.Module):
    def __init__(self, width_multiplier=1.0, architecture=ARCH, num_classes=1000):
        super(MobileNetV2, self).__init__()
        for i in range(len(architecture)):
            architecture[i][1] = int(architecture[i][1]*width_multiplier)

        self.feature, feature_channels = self.make_feature(architecture)
        self.classifier = self.make_classifier(feature_channels, num_classes)

    def forward(self, x):
        return self.classifier(self.feature(x))

    def make_block(self, seq, in_channels):
        t, c, n, s = seq
        if t is None:
            if s == 2:
                return nn.Conv2d(in_channels, c, kernel_size=3, stride=2, padding=1), c
            else:
                return nn.Conv2d(in_channels, c, kernel_size=1), c
        layers = [InvertedResidualLinearBottleneck(in_channels, c, t, s)]
        for _ in range(n-1):
            layers.append(InvertedResidualLinearBottleneck(c, c, t, 1))

        return nn.Sequential(*layers), c

    def make_feature(self, architecture, in_channels=3):
        feature = []
        for seq in architecture:
            block, in_channels = self.make_block(seq, in_channels)
            feature.append(block)

        return nn.Sequential(*feature), in_channels

    def make_classifier(self, feature_channels, num_classes):
        return nn.Sequential(nn.AdaptiveAvgPool2d(1),
                             Flatten(),
                             nn.Linear(feature_channels, num_classes))


def mobilenet_v2_x1_0(**kwargs):
    return MobileNetV2(width_multiplier=1.0, **kwargs)


def mobilenet_v2_x0_5(**kwargs):
    return MobileNetV2(width_multiplier=0.5, **kwargs)


def mobilenet_v2_x2_0(**kwargs):
    return MobileNetV2(width_multiplier=2.0, **kwargs)






