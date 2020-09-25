import numpy as np
import torch as t
import detect.yolo_detect as yolo_detect
import detect.decode as decode
from utils.config import Config as cfg
import utils.utils as utils


class Yolo():
    def __init__(self):
        super().__init__()
        self.Detection = yolo_detect.YoloBody()
        self.Decode = decode.Decodebox(cfg["yolo"]["anchor"], cfg["class_num"], cfg["img_size"])
        self.conf = cfg["confidence"]

    def detect(self, img):
        
        img2 = utils.pred_img(img)
        
        resize_img = np.array(img2, dtype=np.float32)
        resize_img /= 255.0
        resize_img = np.transpose(resize_img, (1, 2, 0))
        # resize_img = np.astype(np.float32) 
        images = []
        images.append(resize_img)
        images = np.asarray(images)
        images = images.transpose((0, 3, 1, 2))
        images = t.from_numpy(images)

        outputlist = []
        if t.cuda.is_available:
            print("cuda is on")
            self.Detection.cuda()
            images = images.cuda()
            out = self.Detection(images)
            for i in range(3):
                outputlist.append(self.Decode(out[i]))
                # outputlist.append(self.Decode(out[i]))
        else:
            print("cuda is off")
            out = self.Detection(images)
            for i in range(3):
                outputlist.append(self.Decode(out[i]))
        for i in outputlist:
            print(i.shape)
        output = t.cat(outputlist, 1)

        betch_detection = utils.nms(output, cfg["class_num"], conf_thres=self.conf, nms_thres=0.4)
        print(output[0,0,0])
        return output