import sys
sys.path.append('C:\\Users\\10376\\Desktop\\大创')
import torch as t
import torch.nn as nn
import torchvision as tv
from yolov3.detect.yolo_Darknet53 import darknet53
import time
import cv2 
import numpy as np

def conv(in_channel, out_channel, ksize):
    pad = (ksize - 1)//2 if ksize else 0
    return  nn.Sequential(
        nn.Conv2d(in_channel, out_channel, ksize, stride = 1, padding = pad),
        nn.BatchNorm2d(out_channel),
        nn.LeakyReLU(0.1)
    )

def YoloConv(in_channel, list_channel , out_channel):

        conv_list = [conv(in_channel, list_channel[0],  1),
            conv(list_channel[0],  list_channel[1],   3),
            conv(list_channel[1],  list_channel[0],   1),
            conv(list_channel[0],  list_channel[1],   3),
            conv(list_channel[1],  list_channel[0],   1),
            conv(list_channel[0],  list_channel[1],   3),
            nn.Conv2d(list_channel[1], out_channel, kernel_size = 1, stride = 1, padding=0, bias=True)]

        layer = nn.ModuleList(conv_list)
        return layer


class YoloBody(nn.Module):
    def __init__(self):
        super(YoloBody, self).__init__()

        self.backbone = darknet53(None)

        # self.config = 
        self.layer1_list = YoloConv(384, [128, 256], 75)

        self.layer2_list = YoloConv(768, [256, 512], 75)
        self.layer2_conv = conv(256, 128, 1)
        self.layer2_Upsample = nn.Upsample(scale_factor=2, mode = 'nearest')

        self.layer3_list = YoloConv(1024, [512, 1024], 75)
        self.layer3_conv = conv(512, 256, 1)
        self.layer3_Upsample = nn.Upsample(scale_factor=2, mode = 'nearest')

    def forward(self, x):
        def get_layer(layer_list, ft_layer):
            for i, layer in enumerate(layer_list):
                ft_layer = layer(ft_layer)
                if i == 4:
                    out_layer = ft_layer
            return ft_layer, out_layer
        #ft3 = 13*13*1024 
        #ft2 = 26*26*512
        #ft1 = 52*52*256
        ft_layer1, ft_layer2, ft_layer3 = self.backbone(x)

        out3, out_layer3 = get_layer(self.layer3_list, ft_layer3)
        out_layer3 = self.layer3_conv(out_layer3)
        out_layer3 = self.layer3_Upsample(out_layer3)
        out_layer3 = t.cat([out_layer3, ft_layer2], 1)

        out2, out_layer2 = get_layer(self.layer2_list, out_layer3)
        out_layer2 = self.layer2_conv(out_layer2)
        out_layer2 = self.layer2_Upsample(out_layer2)
        out_layer2 = t.cat([out_layer2, ft_layer1], 1)

        out1, out_layer1 = get_layer(self.layer1_list, out_layer2)
        #out1 = 52*52*75
        #out2 = 26*26*75
        #out3 = 13*13*75
        return out1, out2, out3





if __name__ == '__main__':
    yolo = YoloBody()

    x = t.Tensor(1, 3, 416, 416)
    print(x.numel())
    start = time.time()
    out = yolo(x)
    end = time.time()
    # print(len(out))
    # for i in out:
    # output = t.cat(out)
    print(end-start)
    print('================')
    yolo2 = YoloBody().cuda().half()
    y = t.cuda.HalfTensor(1, 3, 416, 416)
    start = time.time()
    out2 = yolo2(y)
    end = time.time()
    print(end-start)
    print('================')
