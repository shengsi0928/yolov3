import sys
sys.path.append('C:\\Users\\10376\\Desktop\\大创\\yolov3')
from detect.yolo_detect import YoloBody
import torch as t
import time 
import numpy as np


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    yolo = YoloBody()

    x = t.Tensor(1, 3, 416, 416)
    start = time.time()
    out1, out2, out3 = yolo(x)
    print(out1.size(), out2.size(), out3.size())
    resize_out1 = out1.reshape(1, 3, 25, 52, 52)
    resize_out2 = out2.reshape(1, 3, 25, 26, 26)
    resize_out3 = out3.reshape(1, 3, 25, 13, 13)
    out = resize_out1.detach().numpy()
    print(out[0][0][0])
    end = time.time()
    print(end-start)
    print('================')
