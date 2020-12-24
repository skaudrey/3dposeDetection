"""
__config__ contains the options for training and testing
Basically all of the variables related to training are put in __config__['train'] 
"""
import torch
from torch import nn
import os
from torch.nn import DataParallel
import importlib
import importlib

__config__ = {
    'network': 'model.hourglass.posenet.PoseNet',
    'inference': {
        'nstack': 8,
        'inp_dim': 256, # 256
        'oup_dim': 16,
        'num_parts': 16,
        'increase': 0,
        'keys': ['imgs'],
        'num_eval': 1, ## number of val examples used. entire set is 2958
        # 'train_num_eval': 20, ## number of train examples tested at test time
    },
}
def make_input(t, requires_grad=False, need_cuda = True):
    inp = torch.autograd.Variable(t, requires_grad=requires_grad)
    if need_cuda:
        inp = inp.cuda()
    return inp

def make_output(x):
    if not (type(x) is list):
        return x.cpu().data.numpy()
    else:
        return [make_output(i) for i in x]


def importNet(net):
    t = net.split('.')
    path, name = '.'.join(t[:-1]), t[-1]
    module = importlib.import_module(path)
    return eval('module.{}'.format(name))

class Trainer(nn.Module):
    """
    The wrapper module that will behave differetly for training or testing
    inference_keys specify the inputs for inference
    """
    def __init__(self, model, inference_keys, calc_loss=None):
        super(Trainer, self).__init__()
        self.model = model
        self.keys = inference_keys
        self.calc_loss = calc_loss

    def forward(self, imgs, **inputs):
        inps = {}
        labels = {}

        for i in inputs:
            if i in self.keys:
                inps[i] = inputs[i]
            else:
                labels[i] = inputs[i]

        if not self.training:
            return self.model(imgs, **inps)
        else:
            combined_hm_preds = self.model(imgs, **inps)
            if type(combined_hm_preds)!=list and type(combined_hm_preds)!=tuple:
                combined_hm_preds = [combined_hm_preds]
            loss = self.calc_loss(**labels, combined_hm_preds=combined_hm_preds)
            return list(combined_hm_preds) + list([loss])

def make_network(configs):
    # train_cfg = configs['train']
    config = configs['inference']

    def calc_loss(*args, **kwargs):
        return poseNet.calc_loss(*args, **kwargs)
    
    ## creating new posenet
    PoseNet = importNet(configs['network'])
    poseNet = PoseNet(**config)
    forward_net = DataParallel(poseNet.cuda())
    config['net'] = Trainer(forward_net, configs['inference']['keys'], calc_loss)

    def make_train(batch_id, config, phase, **inputs):
        for i in inputs:
            try:
                inputs[i] = make_input(inputs[i])
            except:
                pass #for last input, which is a string (id_)
                
        net = config['inference']['net']
        config['batch_id'] = batch_id

        # net = net.train() # to resume training

        out = {}
        net = net.eval()
        result = net(**inputs)
        if type(result)!=list and type(result)!=tuple:
            result = [result]
        out['preds'] = [make_output(i) for i in result]
        return out
    return make_train
