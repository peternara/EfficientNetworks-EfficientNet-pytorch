import torch
from torch import nn
from torch.nn import functional as F


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvBnReLU(nn.Module):
    def __init__(self, **kwargs):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(**kwargs)
        self.bn = nn.BatchNorm2d(self.conv.out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DepthwiseSeparableConvolution2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(DepthwiseSeparableConvolution2d, self).__init__()
        self.depthwise = ConvBnReLU(in_channels=in_channels, out_channels=in_channels,
                                    kernel_size=kernel_size, stride=stride, padding=1,
                                    groups=in_channels)
        self.pointwise = ConvBnReLU(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class MobileNetV1(nn.Module):
    def __init__(self, width_multiplier=1.0, base_channels=32,
                 num_layers_per_block=[2, 2, 2, 6, 1],
                 num_classes=1000):
        super(MobileNetV1, self).__init__()
        base_channels = int(base_channels*width_multiplier)
        self.feature, feature_channels = self.make_feature(base_channels, num_layers_per_block)
        self.classifier = self.make_classifier(feature_channels, num_classes)

    def forward(self, x):
        return self.classifier(self.feature(x))

    def make_block(self, in_channels, num_layers):
        layers = []
        for _ in range(num_layers-1):
            layers.append(DepthwiseSeparableConvolution2d(in_channels, in_channels))
        layers.append(DepthwiseSeparableConvolution2d(in_channels, in_channels*2,
                                                      kernel_size=3, stride=2))
        return nn.Sequential(*layers)

    def make_feature(self, base_channels, num_layers_per_block):
        features = []
        features.append(ConvBnReLU(in_channels=3, out_channels=base_channels,
                                   kernel_size=3, stride=2, padding=1))
        for num_layers in num_layers_per_block:
            features.append(self.make_block(base_channels, num_layers))
            base_channels *= 2

        return nn.Sequential(*features), base_channels

    def make_classifier(self, features_channels, num_classes):
        return nn.Sequential(nn.AdaptiveAvgPool2d(1),
                             Flatten(),
                             nn.Linear(features_channels, num_classes))


def mobilenet_v1_x1_0(**kwargs):
    return MobileNetV1(width_multiplier=1.0, **kwargs)


def mobilenet_v1_x0_5(**kwargs):
    return MobileNetV1(width_multiplier=0.5, **kwargs)


def mobilenet_v1_x2_0(**kwargs):
    return MobileNetV1(width_multiplier=2.0, **kwargs)
