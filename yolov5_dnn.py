import random

import cv2
import numpy as np
import time
import math
import os
from numpy import array


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


class yolov5():
    def __init__(self, onnx_path, confThreshold=0.25, nmsThreshold=0.45):
        self.classes = ['pointer', 'nut']
        self.colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(self.classes))]
        num_classes = len(self.classes)
        self.anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.nl = len(self.anchors)
        self.na = len(self.anchors[0]) // 2
        self.no = num_classes + 5
        self.stride = np.array([8., 16., 32.])
        self.inpWidth = 640
        self.inpHeight = 640
        self.net = cv2.dnn.readNetFromONNX(onnx_path)

        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True,
                  stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)

    def box_area(self, boxes: array):
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    def box_iou(self, box1: array, box2: array):
        """
        :param box1: [N, 4]
        :param box2: [M, 4]
        :return: [N, M]
        """
        area1 = self.box_area(box1)  # N
        area2 = self.box_area(box2)  # M
        # broadcasting, 两个数组各维度大小 从后往前对比一致， 或者 有一维度值为1；
        lt = np.maximum(box1[:, np.newaxis, :2], box2[:, :2])
        rb = np.minimum(box1[:, np.newaxis, 2:], box2[:, 2:])
        wh = rb - lt
        wh = np.maximum(0, wh)  # [N, M, 2]
        inter = wh[:, :, 0] * wh[:, :, 1]
        iou = inter / (area1[:, np.newaxis] + area2 - inter)
        return iou  # NxM

    def numpy_nms(self, boxes: array, scores: array, iou_threshold: float):

        idxs = scores.argsort()  # 按分数 降序排列的索引 [N]
        keep = []
        while idxs.size > 0:  # 统计数组中元素的个数
            max_score_index = idxs[-1]
            max_score_box = boxes[max_score_index][None, :]
            keep.append(max_score_index)

            if idxs.size == 1:
                break
            idxs = idxs[:-1]  # 将得分最大框 从索引中删除； 剩余索引对应的框 和 得分最大框 计算IoU；
            other_boxes = boxes[idxs]  # [?, 4]
            ious = self.box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
            idxs = idxs[ious[0] <= iou_threshold]

        keep = np.array(keep)
        return keep

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def non_max_suppression(self, prediction, conf_thres=0.25, agnostic=False):  # 25200 = 20*20*3 + 40*40*3 + 80*80*3
        xc = prediction[
                 ..., 4] > conf_thres  # candidates,获取置信度，prediction为所有的预测结果.shape(1, 25200, 21),batch为1，25200个预测结果，21 = x,y,w,h,c + class个数
        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        output = [np.zeros((0, 6))] * prediction.shape[0]
        # for p in prediction:
        #     for i in p:
        #         with open('./result.txt','a') as f:
        #             f.write(str(i) + '\n')
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            x = x[xc[xi]]  # confidence，获取confidence大于conf_thres的结果
            if not x.shape[0]:
                continue
            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])
            # Detections matrix nx6 (xyxy, conf, cls)
            conf = np.max(x[:, 5:], axis=1)  # 获取类别最高的置信度
            j = np.argmax(x[:, 5:], axis=1)  # 获取下标
            # 转为array：  x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            re = np.array(conf.reshape(-1) > conf_thres)
            # 转为维度
            conf = conf.reshape(-1, 1)
            j = j.reshape(-1, 1)
            # numpy的拼接
            x = np.concatenate((box, conf, j), axis=1)[re]
            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = self.numpy_nms(boxes, scores, self.nmsThreshold)
            output[xi] = x[i]
        return output

    def detect(self, src_img):
        im = src_img.copy()
        im, ratio, wh = self.letterbox(src_img, self.inpWidth, stride=self.stride, auto=False)
        # Sets the input to the network
        blob = cv2.dnn.blobFromImage(im, 1 / 255.0, swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())[0]
        # NMS
        pred = self.non_max_suppression(outs, self.confThreshold, agnostic=False)

        center_nut = [100000000, 100000000]  # 表示inf
        point = [0, 0]

        # draw box
        for i in pred[0]:
            left = int((i[0] - wh[0]) / ratio[0])
            top = int((i[1] - wh[1]) / ratio[1])
            width = int((i[2] - wh[0]) / ratio[0])
            height = int((i[3] - wh[1]) / ratio[1])
            conf = i[4]
            classId = i[5]
            _class = self.classes[int(classId)]
            cv2.rectangle(src_img, (int(left), int(top)), (int(width), int(height)), colors(classId, True), 5,
                          lineType=cv2.LINE_AA)
            # label = '%.2f' % conf
            # label = '%s:%s' % (self.classes[int(classId)], label)

            label = '%s' % (self.classes[int(classId)])

            center_x = (int(left) + int(width)) // 2
            center_y = (int(top) + int(height)) // 2

            # print(f'{_class}:\tloc={center_x, center_y}')

            if _class == 'nut' and center_nut[1] > center_y:
                center_nut[0] = center_x
                center_nut[1] = center_y
            if _class == 'pointer':
                # pointer = src_img[top + 5:height - 5, left + 5:width - 5]  # 这个加5减5是为了去除框的红线的，虽然其实可以在画红线前取
                # cv2.imshow("1", pointer)
                # cv2.imwrite(f"./pointer/{random.randint(1, 20)}.jpg", pointer)
                point[0] = center_x
                point[1] = center_y

            # Display the label at the top of the bounding box
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            cv2.putText(src_img, label, (int(left - 20), int(top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255),
                        thickness=4, lineType=cv2.LINE_AA)

        fix_center_nut_y = center_nut[1] - 130  # 修正后的中心y坐标

        cv2.line(src_img, (point[0], point[1]), (center_nut[0], fix_center_nut_y), (0, 255, 0), 5)  # 画指针线

        cv2.line(src_img, (center_nut[0], 0), (center_nut[0], src_img.shape[0]), (255, 255, 0), 5)  # 中心y轴
        cv2.line(src_img, (0, fix_center_nut_y), (src_img.shape[1], fix_center_nut_y), (255, 255, 0), 5)  # 中心x轴

        a = math.radians(90)  # 旋转到左边-0.1 (-45度)
        r_x = (center_nut[0] - center_nut[0]) * math.cos(a) - (fix_center_nut_y - src_img.shape[0]) * math.sin(-a) + \
              center_nut[0]
        r_y = (center_nut[0] - center_nut[0]) * math.sin(-a) + (fix_center_nut_y - src_img.shape[0]) * math.cos(a) + \
              src_img.shape[0]

        cv2.line(src_img, (center_nut[0], fix_center_nut_y), (int(r_x), int(r_y)), (255, 255, 0), 5)

        b = math.radians(90)  # 旋转到右边0.9 (-45度)
        r_x = (center_nut[0] - center_nut[0]) * math.cos(b) - (fix_center_nut_y - src_img.shape[0]) * math.sin(b) + \
              center_nut[0]
        r_y = (center_nut[0] - center_nut[0]) * math.sin(b) + (fix_center_nut_y - src_img.shape[0]) * math.cos(b) + \
              src_img.shape[0]

        # print(r_x, r_y)

        cv2.line(src_img, (center_nut[0], fix_center_nut_y), (int(r_x), int(r_y)), (255, 255, 0), 5)

        a = [r_x - center_nut[0], r_y - fix_center_nut_y]  # 重新标定以表盘中心为原点的坐标

        b = [point[0] - center_nut[0], point[1] - fix_center_nut_y]  # 重新标定以表盘中心为原点的坐标

        beta = math.acos(  # ab=|a||b|cos(a)
            (a[0] * b[0] + a[1] * b[1]) / (math.sqrt(a[0] ** 2 + a[1] ** 2) * math.sqrt(b[0] ** 2 + b[1] ** 2)))

        # print(a, b)
        print(f'beta={beta * 180 / math.pi}')

        print(f'center_nut = {center_nut[0], fix_center_nut_y}')
        print(f'pointer = {point[0], point[1]}')

        k = -(fix_center_nut_y - point[1]) / (center_nut[0] - point[0])
        print('k = %f' % k)

        eps = math.radians(45) - math.atan(k)  # (180 - 98) / 2

        ra = 1 / math.radians(270)
        num = ra * eps - 0.1
        print(f'num = {num}')

        cv2.putText(src_img, f'num={num:.5f}', (point[0], point[1] + 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0),
                    thickness=3, lineType=cv2.LINE_AA)

        return src_img


def mult_test(onnx_path, img_dir, save_root_path, video=False):
    model = yolov5(onnx_path)
    if video:
        cap = cv2.VideoCapture(0)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)  # 视频平均帧率
        size = (frame_height, frame_width)  # 尺寸和帧率和原视频相同
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('zi.mp4', fourcc, fps, size)
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = model.detect(frame)
            out.write(frame)
            cv2.imshow('result', frame)
            c = cv2.waitKey(1) & 0xFF
            if c == 27 or c == ord('q'):
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    else:
        if not os.path.exists(save_root_path):
            os.mkdir(save_root_path)
        for root, dir, files in os.walk(img_dir):
            for file in files:
                image_path = os.path.join(root, file)
                save_path = os.path.join(save_root_path, file)
                if "mp4" in file or 'avi' in file:
                    cap = cv2.VideoCapture(image_path)
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    size = (frame_width, frame_height)
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(save_path, fourcc, fps, size)
                    while cap.isOpened():
                        ok, frame = cap.read()
                        if not ok:
                            break
                        frame = model.detect(frame)
                        out.write(frame)
                    cap.release()
                    out.release()
                    print("  finish:   ", file)
                elif 'jpg' or 'png' in file:
                    srcimg = cv2.imread(image_path)
                    srcimg = model.detect(srcimg)
                    print(f"finish:   {file}\n")
                    cv2.imwrite(save_path, srcimg)
