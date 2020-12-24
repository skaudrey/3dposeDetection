#!/usr/bin/env python
# encoding: utf-8
'''
@author: Miao Feng
@contact: skaudreymia@gmail.com
@software: PyCharm
@file: yolo.py
@time: 2020/12/19 19:14
@desc: Reference: https://github.com/luaffjk/Image-Processing/blob/master/detecting.ipynb
'''
import cv2
import os
import numpy as np
# from utils.img import compute_iou

class YOLO(object):
    def __init__(self,model_path = '%s/data/model/yolo'%os.getcwd()):
        self.model = cv2.dnn.readNet('%s/yolov3.weights'%model_path,
                                     '%s/yolov3.cfg'%model_path)

        self.classes = self._read_classes('%s/coco.names'%model_path)
        self.thershold = 0.5 # only pick boxes that with more than 50% confidence.

    def _read_classes(self,filename):
        classes = None
        with open(filename, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    def predBox(self,img):
        Width = img.shape[1]
        Height = img.shape[0]
        self.model.setInput(cv2.dnn.blobFromImage(img, 0.00392, (800, 800), (0, 0, 0), True, crop=False))
        layer_names = self.model.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.model.getUnconnectedOutLayers()]
        outs = self.model.forward(output_layers)

        confidences = []
        boxes = []
        # create bounding box
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.thershold:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    label = str(self.classes[class_id])
                    if label=='person': # pick out person's box
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

        return self._getcs(boxes)

    def _getcs(self,bounding):
        '''
        Compute bouding box, organized as  c = (#boxes,2), s=(#boxes,1)
        :param bounding: person's bounding box, shape = (#boxes,4), each item indicate an item (x,y,w,h)
        :return:
        '''
        c,s = [],[]
        for box in bounding:
            x, y, w, h = box
            center = np.mean(np.array([[x, y], [x + w, y + h]]), axis=0)
            scale = h / 200
            c.append(center.tolist())
            s.append(scale)

        print("yolov3--> center = ", c, "  scale = ", s)
        return c, s

    # def _select_box(self,boxes):
    #     # now it only consider single person,can be changed for multiple person
    #     result = None
    #     for i in range(len(boxes)):
    #         box = boxes[i]
    #         rect = (box[0],box[1],box[2]+box[0],box[3]+box[1])
    #         if(compute_iou(rect,self.raw_rect))>0.5:
    #             result = list(self.raw_rect)
    #         else:
    #             result = box
    #     return result




