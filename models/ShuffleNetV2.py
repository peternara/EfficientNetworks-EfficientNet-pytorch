import torch
from torch import nn
# from torch.nn import functional as F
# from ShuffleNetV1 import ChannelShuffle
# from MobileNetV1 import Flatten

# Equal channel width minimize memory access cost
# Excessive group convolution increases MAC
# Network fragmentation reduces degree of parallelism
# Element-wise operations are non-negligible


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


class BasicUnit(nn.Module):
    def __init__(self, in_channels):
        super(BasicUnit, self).__init__()
        right_channels = in_channels//2
        self.right_branch = nn.Sequential(
            nn.Conv2d(right_channels, right_channels, kernel_size=1),
            nn.BatchNorm2d(right_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(right_channels, right_channels,
                      kernel_size=3, stride=1, padding=1,
                      groups=right_channels),
            nn.BatchNorm2d(right_channels),

            nn.Conv2d(right_channels, right_channels, kernel_size=1),
            nn.BatchNorm2d(right_channels),
            nn.ReLU(inplace=True),
        )

        self.shuffle = ChannelShuffle(groups=2)

    def forward(self, x):
        left_x, right_x = x.chunk(2, dim=1)
        right_x = self.right_branch(right_x)

        x = torch.cat([left_x, right_x], dim=1)
        return self.shuffle(x)


class DownsampleUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleUnit, self).__init__()
        right_channels = out_channels//2
        left_channels = out_channels//2

        self.right_branch = nn.Sequential(
            nn.Conv2d(in_channels, right_channels, kernel_size=1),
            nn.BatchNorm2d(right_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(right_channels, right_channels,
                      kernel_size=3, stride=2, padding=1,
                      groups=right_channels),
            nn.BatchNorm2d(right_channels),

            nn.Conv2d(right_channels, right_channels, kernel_size=1),
            nn.BatchNorm2d(right_channels),
            nn.ReLU(inplace=True),
        )

        self.left_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, stride=2, padding=1,
                      groups=in_channels),
            nn.BatchNorm2d(in_channels),

            nn.Conv2d(in_channels, left_channels, kernel_size=1),
            nn.BatchNorm2d(left_channels),
            nn.ReLU(inplace=True),
        )

        self.shuffle = ChannelShuffle(groups=2)

    def forward(self, x):
        x = torch.cat([self.left_branch(x), self.right_branch(x)], dim=1)
        return self.shuffle(x)


complexity_architecture_map = {
    0.5: {"head_channel": 24,
          "base_channel": 48,
          "s2repeat": 3,
          "s3repeat": 7,
          "s4repeat": 3,
          "feature_channel": 1024},
    1.0: {"head_channel": 24,
          "base_channel": 116,
          "s2repeat": 3,
          "s3repeat": 7,
          "s4repeat": 3,
          "feature_channel": 1024},
    1.5: {"head_channel": 24,
          "base_channel": 176,
          "s2repeat": 3,
          "s3repeat": 7,
          "s4repeat": 3,
          "feature_channel": 1024},
    2.0: {"head_channel": 24,
          "base_channel": 244,
          "s2repeat": 3,
          "s3repeat": 7,
          "s4repeat": 3,
          "feature_channel": 2048}
}


class ShuffleNetV2(nn.Module):
    def __init__(self, complexity=1.0, num_classes=1000,
                 arch_dict=complexity_architecture_map):
        super(ShuffleNetV2, self).__init__()
        arch = arch_dict[complexity]
        self.feature = self.make_feature(arch)
        self.classifier = self.make_classifier(arch, num_classes)

    def forward(self, x):
        return self.classifier(self.feature(x))

    def make_stage(self, in_channels, out_channels, repeat):
        layers = [DownsampleUnit(in_channels, out_channels)]
        for _ in range(repeat):
            layers.append(
                BasicUnit(out_channels)
            )
        return nn.Sequential(*layers)

    def make_feature(self, arch):

        stages = [nn.Sequential(
            nn.Conv2d(3, arch["head_channel"], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(arch["head_channel"]),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )]
        in_channels = arch["head_channel"]
        out_channels = arch["base_channel"]
        for i in range(2, 4+1):
            stages.append(self.make_stage(in_channels, out_channels, arch["s{}repeat".format(i)]))
            in_channels = out_channels
            out_channels *= 2

        stages.append(nn.Sequential(
            nn.Conv2d(in_channels, arch["feature_channel"], kernel_size=1),
            nn.BatchNorm2d(arch["feature_channel"]),
            nn.ReLU(inplace=True)
        ))

        return nn.Sequential(*stages)

    def make_classifier(self, arch, num_classes):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(arch["feature_channel"], num_classes)
        )


def shufflenet_v2_x0_5(**kwargs):
    return ShuffleNetV2(complexity=0.5, **kwargs)


def shufflenet_v2_x1_0(**kwargs):
    return ShuffleNetV2(complexity=1.0, **kwargs)


def shufflenet_v2_x1_5(**kwargs):
    return ShuffleNetV2(complexity=1.5, **kwargs)


def shufflenet_v2_x2_0(**kwargs):
    return ShuffleNetV2(complexity=2.0, **kwargs)



