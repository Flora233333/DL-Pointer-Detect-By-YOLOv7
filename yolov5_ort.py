import cv2
import onnxruntime
import argparse
import numpy as np
import time

from utils import letterbox, scale_coords


# print(onnxruntime.get_device())
# print(onnxruntime.__version__)
# print(onnxruntime.get_available_providers())


class Detector():

    def __init__(self, opt):
        super(Detector, self).__init__()
        self.input_name = None
        self.output_name = None
        self.model = None
        self.img_size = opt.img_size
        self.threshold = opt.conf_thres
        self.iou_thres = opt.iou_thres
        self.stride = 1
        self.weights = opt.weights
        self.init_model()
        self.names = ['-0.1', 'label', 'nut', 'pointer', 'start']

    def init_model(self):

        sess = onnxruntime.InferenceSession(self.weights, providers=['CPUExecutionProvider'])  # 加载模型权重
        self.input_name = sess.get_inputs()[0].name  # 获得输入节点
        output_names = []
        for i in range(len(sess.get_outputs())):
            print("output node:", sess.get_outputs()[i].name)
            output_names.append(sess.get_outputs()[i].name)  # 所有的输出节点
        print(output_names)
        self.output_name = sess.get_outputs()[0].name  # 获得输出节点的名称
        print(f"input name {self.input_name}-----output_name {self.output_name}")
        input_shape = sess.get_inputs()[0].shape  # 输入节点形状
        print("input_shape:", input_shape)
        self.model = sess

    def preprocess(self, img):

        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]  # 图片预处理
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img).astype(np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        assert len(img.shape) == 4
        return img0, img

    def detect(self, im):

        img0, img = self.preprocess(im)
        # start = time.clock()
        pred = self.model.run(None, {self.input_name: img})[0]  # 执行推理
        pred = pred.astype(np.float32)
        pred = np.squeeze(pred, axis=0)
        # end = time.clock()
        # print(f'deploy cost:{(end - start) * 100}ms')

        boxes = []
        classIds = []
        confidences = []
        # print(pred.shape)
        for detection in pred:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID] * detection[4]  # 置信度为类别的概率和目标框概率值得乘积

            if confidence > self.threshold:
                box = detection[0:4]
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                classIds.append(classID)
                confidences.append(float(confidence))
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.threshold, self.iou_thres)  # 执行nms算法
        pred_boxes = []
        pred_confes = []
        pred_classes = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                confidence = confidences[i]
                if confidence >= self.threshold:
                    pred_boxes.append(boxes[i])
                    pred_confes.append(confidence)
                    pred_classes.append(classIds[i])
        return im, pred_boxes, pred_confes, pred_classes


def main(opt):
    det = Detector(opt)
    image = cv2.imread(opt.img)
    shape = (det.img_size, det.img_size)

    start = time.perf_counter()
    img, pred_boxes, pred_confes, pred_classes = det.detect(image)
    end = time.perf_counter()
    print(f'deploy cost:{(end - start) * 100}ms')

    if len(pred_boxes) > 0:
        for i, _ in enumerate(pred_boxes):
            box = pred_boxes[i]
            left, top, width, height = box[0], box[1], box[2], box[3]
            box = (left, top, left + width, top + height)
            box = np.squeeze(
                scale_coords(shape, np.expand_dims(box, axis=0).astype("float"), img.shape[:2]).round(), axis=0).astype(
                "int")  # 进行坐标还原
            x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
            # 执行画图函数
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), thickness=2)
            cv2.putText(image, '{0}--{1:.2f}'.format(det.names[pred_classes[i]], pred_confes[i]), (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)
    # cv2.imshow("detector", image)
    # cv2.imwrite('output.jpg', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/web_and_power-station_data.onnx',
                        help='onnx path(s)')
    parser.add_argument('--img', type=str, default='./input_image/new.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    opt, _ = parser.parse_known_args()
    main(opt)
