import torch as t
import numpy as np
import math
import cv2 
def nms(pred_data, num_classes, conf_thres = 0.5, nms_thres = 0.5):
    box_corner = pred_data.new(pred_data.shape)
    best_scores = []
    box_corner[:, :, 0] = pred_data[:, :, 0] - pred_data[:, :, 2] / 2
    box_corner[:, :, 1] = pred_data[:, :, 1] - pred_data[:, :, 3] / 2
    box_corner[:, :, 2] = pred_data[:, :, 0] + pred_data[:, :, 2] / 2
    box_corner[:, :, 3] = pred_data[:, :, 1] + pred_data[:, :, 3] / 2
    pred_data[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(pred_data))]
    for image_i, image_pred in enumerate(pred_data):
        conf_list = (pred_data[ ..., 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_list]

        if not image_pred.size(0):
            continue

        class_conf, class_pred = t.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        print("class_conf:",class_conf, "len(class_conf)", len(class_conf))
        print("class_pred",class_pred, "len(class_pred)", len(class_pred))

        # image_pred[:5]代表了x1, y1, x2, y2, obj_conf,后续与class_conf, class_pred进行重叠
        detections = t.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        #将检测到的种类提取出来 
        unique_labels = detections[:, -1].cpu().unique()

        if pred_data.is_cuda:
            unique_labels = unique_labels.cuda()

        # 检测
        for c in unique_labels:
            detections_class = detections[detections[:, -1] == c]
            
            # 按照检测类型置信度提取前4个预测种类
            _, conf_sort_index = t.sort(detections_class[:4], descending=True)
            detections_class = detections_class(conf_sort_index)
            
            max_detection = []
            while detections_class.size(0):
                # 
                max_detection.append(detections_class[0].unsqueeze(0))
                if len(detections_class) == 1:
                    break;
                ious = bbox_iou(max_detection[-1], detections_class[1:])
                max_detection = detections_class[1:][ious < nms_thres] 
            max_detection = t.cat(max_detection).data
            output[image_i] = max_detection if output[image_i] is None else t.cat(
                (output[image_i], max_detection)
            )
    # print("box_corner.shape:",box_corner.shape)



    return output

    

def pred_normalization(pred_data, img_size):

    FloatTensor = t.cuda.FloatTensor if pred_data.is_cuda else t.FloatTensor
    LongTensor = t.cuda.LongTensor if pred_data.is_cuda else t.LongTensor
    grid_num = int(math.sqrt(pred_data.size(1)/3))



    x = pred_data[..., 0]
    y = pred_data[..., 1]
    w = pred_data[..., 2]
    h = pred_data[..., 3]
    conf = pred_data[..., 4]

    grid_x = t.linspace(0, grid_num-1, steps = grid_num).repeat(grid_num, 1).repeat(
        1 * 3, 1, 1).view(x.shape).type(FloatTensor)
    grid_y = t.linspace(0, grid_num-1, steps = grid_num).repeat(grid_num, 1).t().repeat(
        1 * 3, 1, 1).view(y.shape).type(FloatTensor)

    nor_x = x / img_size[0] * grid_num

    return x


def pred_img(img):
    # img_shape = img.shape[:2]
    
    img = img.squeeze()
    convert_img = cvt_img(img, (416, 416))
    return convert_img



def cvt_img(img, cvt_size):         #为矩形图片添加灰条
    img_h, img_w = img.shape[1:]
    w, h = cvt_size
    # img_copy = img.sq
    resize_x = min(h/img_h, w/img_w)
    new_w = int(img_w * resize_x)
    new_h = int(img_h * resize_x)
    resize_img = img.numpy()
    resize_img = resize_img.transpose((1, 2, 0))
    print("img2:", resize_img.shape)
    resize_img = cv2.resize(resize_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((w, h, 3), 128, np.uint8)
    
    canvas[:, (h-new_h)//2 : (h-new_h)//2 + new_h, (w - new_w)//2 : (w - new_w)//2 + new_w] = resize_img
    canvas = canvas.transpose((2, 1, 0))
    canvas = t.from_numpy(canvas)
    # canvas.unsqueeze_(0)
    return canvas



def bbox_iou(max_detection, detections_class):
    




    return max_detection