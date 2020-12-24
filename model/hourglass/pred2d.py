#!/usr/bin/env python
# encoding: utf-8
'''
@author: Miao Feng
@contact: skaudreymia@gmail.com
@software: PyCharm
@file: pred2d.py
@time: 2020/12/17 22:04
@desc: use pretrained HG model
'''
import os
from datetime import datetime
import time
import importlib
import torch
from model.hourglass.posenet import HeatmapParser
from utils.img import *

parser = HeatmapParser()

PARTS_NAME_JOINT_16 = ['rank', 'rkne', 'rhip','lhip', 'lkne', 'lank','pelv', 'thrx', 'neck', 'head','rwri', 'relb',
                       'rsho','lsho', 'lelb', 'lwri']
flipped_parts = {'mpii':[5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]}

class Detect2D(object):
    def __init__(self,model_path):
        self.func, self.config = self.init(model_path)


    def reload(self, config):
        """
        load or initialize model's parameters by config from config['opt'].continue_exp
        config['train']['epoch'] records the epoch num
        config['inference']['net'] is the model
        """
        model_path = config['model_path']

        # resume = os.path.join('exp', opt.continue_exp)
        resume_file = os.path.join(model_path, 'checkpoint.pt')
        if os.path.isfile(resume_file):
            print("load 8HG model done ")
            checkpoint = torch.load(resume_file)

            config['inference']['net'].load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(model_path))
            exit(0)

    def init(self,model_path):
        """
        task.__config__ contains the variables that control the training and testing
        make_network builds a function which can do forward and backward propagation
        """
        # opt = parse_command_line()
        task = importlib.import_module('model.hourglass.pose')
        config = task.__config__
        config['model_path'] = model_path

        func = task.make_network(config)
        self.reload(config)
        return func, config


    def post_process(self,det, mat_, trainval, c=None, s=None, resolution=None):
        mat = np.linalg.pinv(np.array(mat_).tolist() + [[0, 0, 1]])[:2]
        res = det.shape[1:3]
        cropped_preds = parser.parse(np.float32(
            [det]))  # parse heatmap to joints estimation using the net (mainly maxpooling) in group.py HeatmapParser
        cropped_preds = cropped_preds[0]

        if len(cropped_preds) > 0:
            cropped_preds[:, :, :2] = kpt_affine(cropped_preds[:, :, :2] * 4, mat)  # size 1x16x3

        preds = np.copy(cropped_preds)
        ##for inverting predictions from input res on cropped to original image
        if trainval != 'cropped':
            for j in range(preds.shape[1]):
                preds[0, j, :2] = transform(preds[0, j, :2], c, s, resolution, invert=1)
        return preds

    def inference(self,img, func, config, c = np.array([128,128]),s = 1):
        """
        forward pass at test time
        calls post_process to post process results
        """
        input_res = config['inference']['inp_dim']
        im = crop(img, c, s, (input_res, input_res))

        height, width = im.shape[0:2]
        center = (width / 2, height / 2)
        scale = max(height, width) / 200
        res = (config['inference']['inp_dim'], config['inference']['inp_dim'])

        mat_ = get_transform(center, scale, res)[:2]
        inp = im / 255

        def array2dict(tmp):
            return {
                'det': tmp[0][:, :, :16],
            }

        # predict heatmaps, func([inp]) will run hg model and predict heatmaps
        hm1 = array2dict(func([inp]))  # func[inp] will run hourglass in stack for input data "inp"
        # shape = (1,#stacks,#joints,#output_res, #output_res
        hm2 = array2dict(func([inp[:, ::-1]])) # GBR

        hms = {}
        for ii in hm1:
            hms[ii] = np.concatenate((hm1[ii], hm2[ii]), axis=0)

        det = hms['det'][0, -1] + hms['det'][1, -1, :, :, ::-1][flipped_parts['mpii']]
        if det is None:
            return [], []
        det = det / 2

        det = np.minimum(det, 1)

        return self.post_process(det, mat_, 'valid', c, s, res)


    def pred2d(self,img,c,s):
        '''
        :param img:
        :param c:
        :param s:
        :return: Joints in order ['RFoot','RKnee','RHip','LHip','LKnee','LFoot','Hip','Thorax','Neck','Head','RWrist','RElbow','RShoulder','LShoulder','LElbow','LWrist']
        '''
        img = img[:,:,::-1] # change to RGB

        def runner(imgs):
            return self.func(0, self.config, 'inference', imgs=torch.Tensor(np.float32(imgs)))['preds']

        def do(img, c, s):
            '''
            Predict 2D pose for each person (indicate by each pait of center and scale
            in c and s
            :param img:
            :param c: shape = (#person,2)
            :param s: shape = (#person)
            :return:
            '''
            ans_list = []

            for (c_tmp,s_tmp) in zip(c,s):
                ans = self.inference(img, runner, self.config, c_tmp, s_tmp)
                ## ans has shape N,16,3 (num preds, joints, x/y/visible)
                if len(ans) > 0:
                    ans = ans[:, :, :3]
                # print(type(ans))
                pred = []
                for i in range(ans.shape[0]):
                    pred.append(ans[i, :, :])
                ans_list.append(pred)

            return ans_list

        pred = do(img, c, s)
        pred_2d = []
        for i in pred:
            pred_2d.append(i[0][:,:2])
        pred_2d = np.array(pred_2d)
        print('Predict 2d pose: Done')

        return pred_2d

def print_run_time(func): # It will cause the func won't return it's result
    result = None
    def wrapper(*args, **kw):
        local_time = time.clock()
        result = func(*args, **kw)
        print('Function [%s] spend %.2f seconds' % (func.__name__ ,time.clock() - local_time))
    return wrapper

def getcs(bounding):
    c,s = [],[]
    for i in bounding:
        x,y,w,h = i
        center = np.mean(np.array([[x,y],[x+w,y+h]]),axis=0)
        scale = h/200
        c.append(center.tolist())
        s.append(scale)
    print("yolov3--> center = ", c, "  scale = ", s)
    return c,s


if __name__=='__main__':
    basepath = os.getcwd()
    img_params = [
        {'imgpath':'MPII/images/000003072.jpg',
         # 'box':[[850.5, 133.5, 173, 329],[547.5, 94.5, 193, 357],[1111.5, -12.0, 169, 566]] # 015401864.jpg
         'box': [[356.0, 74.5, 158, 513],[612.5, 113.0, 283, 486]] #000003072.jpg
         },
        {
            'imgpath': 'h36m/imgs/S9_Directions.60457274_000246.jpg',
            # 'box':[[247.0, 116.5, 488, 521]] # S9_Directions.60457274_0000006.jpg
            # 'box': [[244.5, 119.5, 493, 517]] # S9_Directions.60457274_00000026.jpg
            'box':[[589.5, 172.0, 213, 476]] # S9_Directions.60457274_000246.jpg
        },
        {
            'imgpath': 'images/test-1.jpg',
            'box':[[187.5, 110.0, 487, 708]]
        }
    ]

    id = 0
    img = cv2.imread("%s/../../data/%s"%(basepath,img_params[id]['imgpath']))
    print(img.shape)

    hg_model_path = '%s/../../data/model/hg'%basepath
    siamese_model = "%s/../../data/model/siamese" % basepath
    from model.siamese.Pred3D import Detect3D
    from utils.plot import plotkp
    import warnings
    warnings.filterwarnings('ignore')

    @print_run_time
    def predict():
        # initialize detectors
        predictor2d = Detect2D(model_path=hg_model_path)

        # do prediction and calculate time
        pred = predictor2d.pred2d(img, c, s)
        # print(pred)
        del predictor2d
        plotkp(img, pred,bounding,None)
        predictor_3d = Detect3D(siamese_model)
        pred_3d = predictor_3d.predict3Dpose(pred)

        for (k, v) in pred_3d.items():
            print(k, ": ", v)

        return pred_3d


    bounding = img_params[id]['box']
    c, s = getcs(bounding)
    predict()
