import torch
from torch import nn
from torch.nn import functional as F
# from MobileNetV1 import Flatten


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.g = groups

    def forward(self, x):
        bs, c, h, w = x.size()
        x = x.view(bs, self.g, -1, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(bs, -1, h, w)
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups, stride):
        super(ShuffleNetUnit, self).__init__()
        self.use_res = True if stride == 1 else False
        bottleneck_channels = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels,
                               kernel_size=1, groups=groups)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.shuffle = ChannelShuffle(groups)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,
                               kernel_size=3, stride=stride, padding=1,
                               groups=bottleneck_channels)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels,
                               kernel_size=1, groups=groups)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.shuffle(x)
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))

        if self.use_res:
            x += identity
        else:
            identity = F.avg_pool2d(identity, kernel_size=3, stride=2, padding=1)
            x = torch.cat([x, identity], dim=1)

        return F.relu(x)


class FirstUnitInStageOne(nn.Module):
    def __init__(self, in_channels, out_channels, groups, stride):
        super(FirstUnitInStageOne, self).__init__()
        self.use_res = True if stride == 1 else False
        bottleneck_channels = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels,
                               kernel_size=1, groups=1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.shuffle = ChannelShuffle(groups)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,
                               kernel_size=3, stride=stride, padding=1,
                               groups=bottleneck_channels)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels,
                               kernel_size=1, groups=groups)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.shuffle(x)
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))

        if self.use_res:
            x += identity
        else:
            identity = F.avg_pool2d(identity, kernel_size=3, stride=2, padding=1)
            x = torch.cat([x, identity], dim=1)

        return F.relu(x)


group_channel_map ={1: 144, 2: 200, 3: 240, 4: 272, 8: 384}


class ShuffleNetV1(nn.Module):
    def __init__(self, groups=3, scale_factor=1.0,
                 num_classes=1000,
                 num_layers_per_stage=[None, 4, 8, 4]):
        super(ShuffleNetV1, self).__init__()
        head_channel = 24
        base_channels = group_channel_map[groups]
        head_channel = int(head_channel*scale_factor)
        base_channels = int(base_channels*scale_factor)
        self.feature, feature_channels = self.make_feature(groups, head_channel,
                                                           base_channels, num_layers_per_stage)
        self.classifier = self.make_classifier(feature_channels, num_classes)

    def forward(self, x):
        return self.classifier(self.feature(x))

    def make_stage(self, in_channels, out_channels,
                   groups, num_layers,
                   small_in_channels=False):
        if small_in_channels:
            layers = [FirstUnitInStageOne(in_channels, out_channels-in_channels,
                                          groups, 2)]
        else:
            layers = [ShuffleNetUnit(in_channels, out_channels-in_channels,
                                     groups, 2)]
        for _ in range(num_layers-1):
            layers.append(ShuffleNetUnit(out_channels, out_channels,
                                         groups, 1))

        return nn.Sequential(*layers)

    def make_feature(self, groups, head_channels, base_channels, num_layers_per_stage):
        stages =[nn.Sequential(nn.Conv2d(3, head_channels, kernel_size=3, stride=2, padding=1),
                               nn.BatchNorm2d(head_channels),
                               nn.ReLU(),
                               nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]
        in_channels = head_channels
        for num_layers in num_layers_per_stage[1:]:
            if in_channels == head_channels:
                stages.append(self.make_stage(in_channels, base_channels, groups, num_layers, True))
            else:
                stages.append(self.make_stage(in_channels, base_channels, groups, num_layers, False))
            in_channels = base_channels
            base_channels = base_channels * 2

        return nn.Sequential(*stages), in_channels

    def make_classifier(self, feature_channels, num_classes):
        return nn.Sequential(nn.AdaptiveAvgPool2d(1),
                             Flatten(),
                             nn.Linear(feature_channels, num_classes))



def shufflenet_v1_x0_5(**kwargs):
    return ShuffleNetV1(scale_factor=0.5, **kwargs)


def shufflenet_v1_x1_0(**kwargs):
    return ShuffleNetV1(scale_factor=1.0, **kwargs)


def shufflenet_v1_x1_5(**kwargs):
    return ShuffleNetV1(scale_factor=1.5, **kwargs)


def shufflenet_v1_x2_0(**kwargs):
    return ShuffleNetV1(scale_factor=2.0, **kwargs)





