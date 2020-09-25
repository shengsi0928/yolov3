import torch as t
import torch.nn as nn
import sys
sys.path.append('C:\\Users\\10376\\Desktop\\大创\\yolov3')
from utils.config import Config as cfg
import cv2
from utils import utils
class Decodebox(nn.Module):
    def __init__(self, anchor, class_num, origin_size):
        super(Decodebox, self).__init__()
        self.anchor_num = len(anchor)
        self.anchor = anchor
        self.bbox_attr = 5 + class_num
        self.class_num = class_num
        self.origin_size = origin_size

    def forward(self, input):
        print(input.size())
        batch_size = input.size(0)
        input_width = input.size(2)
        input_height= input.size(3)
        stride_x = self.origin_size[0] / input_width
        stride_y = self.origin_size[1] / input_height
        anchors = []
        for i in self.anchor:
            anchors.append(i)
        for anchor in anchors:
            scaled_anchors = [(anchor_width / input_width, anchor_height / input_height) for anchor_width, anchor_height in anchor]
        # print("batch_size", batch_size)
        # print("self.anchor_num", self.anchor_num)
        # print("self.bbox_attr",self.bbox_attr)
        # print("input_height", input_height)
        # print("input_width", input_width)
        resize_input = input.resize(batch_size, self.anchor_num, self.bbox_attr, input_height, input_width).permute(0, 1, 3, 4, 2)
        print("resize_iuput.size = ", resize_input.size())
        x = t.sigmoid(resize_input[..., 0])
        y = t.sigmoid(resize_input[..., 1])
        w = resize_input[..., 2]
        h = resize_input[..., 3]

        conf = t.sigmoid(resize_input[..., 4])
        pred_class = t.sigmoid(resize_input[..., 5:])

        FloatTensor = t.cuda.FloatTensor if x.is_cuda else t.FloatTensor
        LongTensor = t.cuda.LongTensor if x.is_cuda else t.LongTensor

        grid_x = t.linspace(0, input_width-1, steps = input_width).repeat(input_width, 1).repeat(
            batch_size * self.anchor_num, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = t.linspace(0, input_height-1, steps = input_height).repeat(input_height, 1).t().repeat(
            batch_size * self.anchor_num, 1, 1).view(y.shape).type(FloatTensor)

        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

        pred_boxes = FloatTensor(resize_input[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = t.exp(w.data) * anchor_w
        pred_boxes[..., 3] = t.exp(h.data) * anchor_h


        anchor_left = grid_x - anchor_w / 2
        anchor_top = grid_y - anchor_h / 2
        pred_left = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
        pred_top = pred_boxes[..., 1] - pred_boxes[..., 3] / 2

        _scale = t.Tensor([stride_x, stride_y] * 2).type(FloatTensor)
        
        out = t.cat( (pred_boxes.view(batch_size, -1, 4) *_scale, conf.view(batch_size, -1, 1), pred_class.view(batch_size, -1, self.class_num)), -1)
        
        return out.data
        