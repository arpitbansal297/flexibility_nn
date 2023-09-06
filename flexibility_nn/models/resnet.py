import torch.nn as nn
import torch
# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, image_size=32):
        super(ResNet, self).__init__()
        self.in_channels = 64 if image_size > 32 else 16
        self.conv = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False) if image_size > 32 else conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if image_size > 32 else None
        self.layers = self.make_layers(block, layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, num_classes)  # Note the change here too.

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def make_layers(self, block, layers):
        blocks_layers = []
        out_channels = self.in_channels
        for i, num_blocks in enumerate(layers):
            out_channels *= 2 if i != 0 else 1  # Double the number of channels for each layer after the first
            blocks_layers.append(self.make_layer(block, out_channels, num_blocks, stride=2 if i != 0 else 1))  # Apply stride 2 for layers after the first
        return nn.ModuleList(blocks_layers)  # Wrap the list of layers with nn.ModuleList so that PyTorch keeps track of the layers.

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.maxpool:
            x = self.maxpool(x)
        for layer in self.layers:
            x = layer(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNet3(num_classes=10, image_size=32):
    return ResNet(ResidualBlock, [2], num_classes=num_classes, image_size=image_size)


def ResNet9(num_classes=10, image_size=32):
    return ResNet(ResidualBlock, [2, 2, 2], num_classes=num_classes, image_size=image_size)


def ResNet6(num_classes=10, image_size=32):
    return ResNet(ResidualBlock, [2, 2], num_classes=num_classes, image_size=image_size)


def ResNet18X(num_classes=10, image_size=32):
    return ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=num_classes, image_size=image_size)


def ResNet20X(num_classes=10, image_size=32):
    return ResNet(ResidualBlock, [2, 2, 2, 2, 2], num_classes=num_classes, image_size=image_size)


def ResNet22X(num_classes=10, image_size=32):
    return ResNet(ResidualBlock, [2, 2, 2, 2, 2, 2], num_classes=num_classes, image_size=image_size)
