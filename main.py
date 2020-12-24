#!/usr/bin/env python
# encoding: utf-8
'''
@author: Miao Feng
@contact: skaudreymia@gmail.com
@software: PyCharm
@file: main.py
@time: 2020/12/2 14:25
@desc:
'''
import cv2
from model.hourglass import Detect2D
from model.siamese import Detect3D
from model.yolo import YOLO
from utils.plot import plotkp
import numpy as np
import time
import socket
import json
import os

__config__={
    'frame_size':800,
    'capture_sleep':15, # stop for 15s after each capture
    'frame_c': np.array([400,400]),
    'frame_s': 4.,
    'host':'127.0.0.1',
    'port': 25001,
    'hg_inp_dim':256,
    'all': False # whether to load all pretrained model while initializing server
}


def print_run_time(func):
    def wrapper(*args, **kw):
        local_time = time.clock()
        func(*args, **kw)
        print('Function [%s] spend %.2f seconds' % (func.__name__ ,time.clock() - local_time))
    return wrapper


class RealTimePose(object):
    def __init__(self,yolo_model_path = '%s/data/model/yolo',hg_model_path = "%s/data/model/hg",siamese_model_path="%s/data/model/siamese"):

        '''It's better to initialize all detectors in init function, but
        my gpu on laptop is in low capacity, loading all models will cause
        out of memory.'''
        self.isall = __config__['all']
        if self.isall:
            self.yolo = YOLO(yolo_model_path)
            self.detector_2d = Detect2D(hg_model_path)
            self.detector_3d = Detect3D(siamese_model_path)
        else:
            self.yolo_model_path = yolo_model_path%os.getcwd()
            self.hg_model_path = hg_model_path%os.getcwd()
            self.siamese_model_path = siamese_model_path%os.getcwd()

        self.captureDevice = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # captureDevice = camera

        # Check open video success
        if not self.captureDevice.isOpened():
            raise Exception("Could not open video device")

        w,h = __config__['hg_inp_dim'],__config__['hg_inp_dim']

        self.captureDevice.set(cv2.CAP_PROP_FRAME_WIDTH, w)  # set width
        self.captureDevice.set(cv2.CAP_PROP_FRAME_HEIGHT, h)  # set height

        self.server = self.listener_init()

        self.isListen = True  # whether open server
        self.doing = False  # whether start predict pose
        self.counter_capture = 0 # counting how many images have we captured
        self.preds = None # maintain predictions

    def capImage(self):
        """
        Capture frame from video device, and return a frame
        :return: return one frame
        """
        w,h = __config__['frame_size'],__config__['frame_size']
        # print('I am gonna wait for a while')

        for i in range(2): # drop the 1st one
            if i > 0:
                time.sleep(5)
            _, frame = self.captureDevice.read()

        img_hg = cv2.resize(frame, (w,h))
        self.counter_capture += 1
        print('-'*10,'capture the %d-th image'%self.counter_capture,10*'-')
        cv2.imwrite("%s/data/images/test-%d.jpg" % (os.getcwd(),self.counter_capture),img_hg)
        return img_hg

    def showSkeletons(self,img,pred2d):
        plotkp(img,pred2d)

    def listener_init(self):
        server = socket.socket(socket.SOCK_DGRAM)
        server.bind((__config__['host'], __config__['port']))
        server.listen(5)

        self.isListen = True
        print("Start waiting......")

        return server

    @print_run_time
    def detect(self,img):
        if self.isall:
            c,s = self.yolo.predBox(img)
            preds_2d = self.detector_2d.pred2d(img, c, s)
            self.showSkeletons(img, preds_2d)
            self.preds = self.detector_3d.predict3Dpose(preds_2d)
        else:
            detector_box = YOLO(img.shape[:2],self.yolo_model_path)
            c,s = detector_box.predBox(img)

            detector_2d = Detect2D(self.hg_model_path)

            preds_2d = detector_2d.pred2d(img, c, s)
            del detector_2d
            self.showSkeletons(img, preds_2d)
            detector_3d = Detect3D(self.siamese_model_path)
            self.preds = detector_3d.predict3Dpose(preds_2d)
            # print(self.preds)
            del detector_3d


    def start(self):
        """
        starter
        :return:
        """
        while self.isListen: # multiple connections
            conn, addr = self.server.accept()
            print("Request form {}".format(addr))
            self.doing = True
            while self.doing: # multiple capture request from one client
                data = conn.recv(1024)
                print("recv:", data)

                if not data:
                    print("client has lost...")
                    break
                if data.decode('utf-8') == 'capture':
                    time.sleep(15)

                    # conn.send(str("Start capture image").encode('utf-8'))
                    img = self.capImage()
                    print('capture one frame done')
                    # conn.send(str("Predicting, please wait around 50 seconds").encode('utf-8'))
                    self.detect(img)
                    # print(type(self.preds))

                    print("prediction done")
                    print("send-->")
                    for (k, v) in self.preds.items():
                        print(k, ": ", v)

                    # preds_send = json.dumps(preds, sort_keys=False)

                    conn.send(str(self.preds).encode('utf-8'))
                    # time.sleep(60)
                elif data.decode('utf-8') == 'break': # only when break, destroy all
                    self.isListen = False
                    self.doing = False
                    print('start exiting')
                    self.exit()
                    break
            break

    def exit(self):
        """
        exit and close server
        :return:
        """
        self.captureDevice.release()
        cv2.destroyAllWindows()

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-h", '--host', help="host", type=str, default='127.0.0.1')
    parser.add_argument("-p", '--port', help="port of service", default=20001, type=int)
    parser.add_argument("-all", '--all', help="load all models while server initialize", default=False, type=bool)
    args = parser.parse_args()

    host, port = args.host, args.port
    __config__['host'], __config__['port'] = host, port
    __config__['all'] = args.all


if __name__=='__main__':
    import warnings
    warnings.filterwarnings('ignore')
    import argparse
    parse_command_line()
    tool = RealTimePose()
    tool.start()