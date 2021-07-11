import torch
import torch.nn as nn

class Resnet(nn.Module):
    def __init__(self):
        super().__init__(self)

        self.prep_layer = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv1 = nn.Conv2d(128, 128, 3, 1, 1) 

        self.layer1 = self.get_resnet_layers(128, 128)
    
    
    def get_resnet_layers(self, in_channels, out_channels):

        layers = []
        layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__(self)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)

        return x



block = Block()

model = Resnet()