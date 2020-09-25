import sys
sys.path.append('C:\\Users\\10376\\Desktop\\大创\\yolov3')
from detect.yolo import Yolo
import torch as t
import time 
import numpy as np
from PIL import Image
import cv2
def func():
    np.set_printoptions(suppress=True)
    x = t.Tensor(1, 3, 416, 416)
    start = time.time()
    yolo = Yolo()
    # try:
    #     img = Image()
    # except:
    #     print("图片打开失败。")
    # else:
    out = yolo.detect(x)
    out2 = out.cpu()
    out2 = np.array(out2, np.uint8)
    # print(out2.shape)
    # cv2.imshow("out", out2)
    end = time.time()
    print(end-start)
    print('================')


if __name__ == '__main__':
    func()
