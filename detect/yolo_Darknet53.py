import torch as t
import torchvision as tv
import torch.nn as nn
import time


class DarknetConv2D(nn.Module):
    def __init__(self, in_features, out_features, ksize = 3, stride = 1, padding = 0):
        super(DarknetConv2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_features, out_features, ksize, stride, padding),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            DarknetConv2D(out_features, out_features//2, ksize=1, stride=1, padding=0),
            DarknetConv2D(out_features//2, out_features, ksize=3, stride=1, padding=1)
        )

    def forward(self, x):
        return x + self.left(x)

class Darknet(nn.Module):

    def __init__(self):
        super(Darknet, self).__init__()
        self.layers_num = [1, 2, 8, 8, 4]
        self.layers_out_ksize = [256, 512, 1024]
        self.Conv_layer = nn.Sequential(
            DarknetConv2D(3, 32, ksize = 3, stride = 1, padding = 1),
            nn.Conv2d(32, 64, 3, stride = 2, padding = 1)
        )
        # self.layer1 = self._make_layer(32, 64, layers_num[0])
        self.layer2 = self._make_layer(64, 128, self.layers_num[1])
        self.layer3 = self._make_layer(128, 256, self.layers_num[2])
        self.layer4 = self._make_layer(256, 512, self.layers_num[3])
        self.layer5 = self._make_layer(512, 1024, self.layers_num[4])


    def _make_layer(self, in_features, out_features, layer_num):
        layers = []
        layers.append(nn.Conv2d(in_features, out_features, 3, stride = 2, padding = 1))
        for i in range(0, layer_num):
            layers.append(ResidualBlock(out_features, out_features))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.Conv_layer(x)
        # x = self.layer1(x)
        x = self.layer2(x)
        detection_52 = self.layer3(x)
        detection_26 = self.layer4(detection_52)
        detection_13 = self.layer5(detection_26)
        return detection_52, detection_26, detection_13

def darknet53(pretrained, **kwargs):
    model = Darknet()
    return model

