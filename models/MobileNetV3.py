import torch
from torch import nn
from torch.nn import functional as F


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class HardSigmoid(nn.Module):
    def __init__(self,inplace=True):
        super(HardSigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu6(x+3)/6


def hardsigmoid(x, inplace=True):
    return F.relu6(x+3, inplace=inplace)/6


class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.hardsigmoid = HardSigmoid(inplace)

    def forward(self, x):
        return x * self.hardsigmoid(x)


def hardswish(x, inplace=True):
    return x * hardsigmoid(x, inplace)


def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups),
        nn.BatchNorm2d(out_channels)
    )


def get_act(non_linear):
    return nn.ReLU(inplace=True) if non_linear == "RE" else HardSwish(inplace=True)


def get_padding(kernel_size):
    return 1 if kernel_size == 3 else 2


class BasicUnit(nn.Module):
    def __init__(self, in_channels, out_channels, exp_size, kernel_size, non_linear):
        super(BasicUnit, self).__init__()
        if in_channels != out_channels:
            raise ValueError
        padding = get_padding(kernel_size)
        self.act = get_act(non_linear)
        self.conv1 = conv_bn(in_channels, exp_size, kernel_size=1)
        self.conv2 = conv_bn(exp_size, exp_size,
                             kernel_size=kernel_size, stride=1, padding=padding, groups=exp_size)
        self.conv3 = conv_bn(exp_size, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)

        return x + identity


class SqueezeeaAndExcite(nn.Module):
    def __init__(self, in_channels):
        super(SqueezeeaAndExcite, self).__init__()
        inter_channels = int(in_channels*0.25)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(inter_channels, in_channels, kernel_size=1)
        self.hsigmoid = HardSigmoid(inplace=True)

    def forward(self, x):
        identity = x
        x = self.pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.hsigmoid(x)
        return x * identity


class SEBasicUnit(nn.Module):
    def __init__(self, in_channels, out_channels, exp_size, kernel_size, non_linear):
        super(SEBasicUnit, self).__init__()
        if in_channels != out_channels:
            raise ValueError
        padding = get_padding(kernel_size)
        self.act = get_act(non_linear)
        self.conv1 = conv_bn(in_channels, exp_size, kernel_size=1)
        self.conv2 = conv_bn(exp_size, exp_size,
                             kernel_size=kernel_size, stride=1, padding=padding, groups=exp_size)
        self.conv3 = conv_bn(exp_size, out_channels, kernel_size=1)

        self.se = SqueezeeaAndExcite(exp_size)

    def forward(self, x):
        identity = x
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.se(x)
        x = self.conv3(x)

        return x + identity


class TransUnit(nn.Module):
    def __init__(self, in_channels, out_channels, exp_size, kernel_size, non_linear):
        super(TransUnit, self).__init__()
        padding = get_padding(kernel_size)
        self.act = get_act(non_linear)
        self.conv1 = conv_bn(in_channels, exp_size, kernel_size=1)
        self.conv2 = conv_bn(exp_size, exp_size,
                             kernel_size=kernel_size, stride=1, padding=padding, groups=exp_size)
        self.conv3 = conv_bn(exp_size, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)

        return x


class SETransUnit(nn.Module):
    def __init__(self, in_channels, out_channels, exp_size, kernel_size, non_linear):
        super(SETransUnit, self).__init__()
        padding = get_padding(kernel_size)
        self.act = get_act(non_linear)
        self.conv1 = conv_bn(in_channels, exp_size, kernel_size=1)
        self.conv2 = conv_bn(exp_size, exp_size,
                             kernel_size=kernel_size, stride=1, padding=padding, groups=exp_size)
        self.conv3 = conv_bn(exp_size, out_channels, kernel_size=1)

        self.se = SqueezeeaAndExcite(exp_size)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.se(x)
        x = self.conv3(x)

        return x


class DownsampleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, exp_size, kernel_size, non_linear):
        super(DownsampleUnit, self).__init__()
        padding = get_padding(kernel_size)
        self.act = get_act(non_linear)
        self.conv1 = conv_bn(in_channels, exp_size, kernel_size=1)
        self.conv2 = conv_bn(exp_size, exp_size,
                             kernel_size=kernel_size, stride=2, padding=padding, groups=exp_size)
        self.conv3 = conv_bn(exp_size, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)

        return x


class SEDownsampleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, exp_size, kernel_size, non_linear):
        super(SEDownsampleUnit, self).__init__()
        padding = get_padding(kernel_size)
        self.act = get_act(non_linear)
        self.conv1 = conv_bn(in_channels, exp_size, kernel_size=1)
        self.conv2 = conv_bn(exp_size, exp_size,
                             kernel_size=kernel_size, stride=2, padding=padding, groups=exp_size)
        self.conv3 = conv_bn(exp_size, out_channels, kernel_size=1)

        self.se = SqueezeeaAndExcite(exp_size)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.se(x)
        x = self.conv3(x)

        return x


class BottleneckBuilder(object):
    def __init__(self):
        super(BottleneckBuilder, self).__init__()

    def build_bneck(self, in_channels, out_channels,
                    exp_size,
                    kernel_size, stride,
                    non_linear, squeeze_and_excite):
        if stride != 1:
            if squeeze_and_excite:
                return SEDownsampleUnit(in_channels, out_channels, exp_size, kernel_size, non_linear)
            else:
                return DownsampleUnit(in_channels, out_channels, exp_size, kernel_size, non_linear)
        elif in_channels != out_channels:
            if squeeze_and_excite:
                return SETransUnit(in_channels, out_channels, exp_size, kernel_size, non_linear)
            else:
                return TransUnit(in_channels, out_channels, exp_size, kernel_size, non_linear)

        else:
            if squeeze_and_excite:
                return SEBasicUnit(in_channels, out_channels, exp_size, kernel_size, non_linear)
            else:
                return BasicUnit(in_channels, out_channels, exp_size, kernel_size, non_linear)


builder = BottleneckBuilder()
large_arch = [
    [3, 16, 16, False, "RE", 1],
    [3, 64, 24, False, "RE", 2],
    [3, 72, 24, False, "RE", 1],
    [5, 72, 40, True, "RE", 2],
    [5, 120, 40, True, "RE", 1],
    [5, 120, 40, True, "RE", 1],
    [3, 240, 80, False, "HS", 2],
    [3, 200, 80, False, "HS", 1],
    [3, 184, 80, False, "HS", 1],
    [3, 184, 80, False, "HS", 1],
    [3, 480, 112, True, "HS", 1],
    [3, 672, 112, True, "HS", 1],
    [5, 672, 160, True, "HS", 2],
    [5, 960, 160, True, "HS", 1],
    [5, 960, 160, True, "HS", 1]
]
small_arch = [
    [3, 16, 16, True, "RE", 2],
    [3, 72, 24, False, "RE", 2],
    [3, 88, 24, False, "RE", 1],
    [5, 96, 40, True, "HS", 2],
    [5, 240, 40, True, "HS", 1],
    [5, 240, 40, True, "HS", 1],
    [5, 120, 48, True, "HS", 1],
    [5, 144, 48, True, "HS", 1],
    [5, 288, 96, True, "HS", 2],
    [5, 576, 96, True, "HS", 1],
    [5, 576, 96, True, "HS", 1]
]


class MobileNetV3(nn.Module):
    def __init__(self, arch, num_classes=1000):
        super(MobileNetV3, self).__init__()
        self.features = self.make_feature(arch)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))

    def make_feature(self, arch):
        layers = [
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            HardSwish(inplace=True),
        ]
        in_channels = 16
        for a in arch:
            kernel_size, exp_size, out_channels, sequeezs_and_excite, non_linear, stride = a
            layers.append(
                builder.build_bneck(in_channels, out_channels,
                                    exp_size,
                                    kernel_size, stride,
                                    sequeezs_and_excite, non_linear)
            )
            in_channels = out_channels

        layers += [nn.Conv2d(in_channels, exp_size, kernel_size=1),
                   nn.BatchNorm2d(exp_size),
                   HardSwish(inplace=True),
                   nn.AdaptiveAvgPool2d(1),
                   nn.Conv2d(exp_size, 1280, kernel_size=1),
                   HardSwish(inplace=True),
                   Flatten()]

        return nn.Sequential(*layers)


def mobilenet_v3_small(**kwargs):
    return MobileNetV3(small_arch, **kwargs)


def mobilenet_v3_large(**kwargs):
    return MobileNetV3(large_arch, **kwargs)


