import torch
import os
import os.path as osp
import random
import numpy as np
from Learning.Losses import SIC_mse_loss, SIC_mae_loss
from Learning.metrics import SIE_Metric

import logging
import os
import sys
from tqdm import tqdm
import pickle as pkl

import time



def set_random_seed(seed):
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    base_seed = torch.IntTensor(1).random_().item()
    # print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)


def logger(name, save_dir, comment=''):
    '''
    Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
    Modified by Jiayuan Gu
    '''
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        filename = 'log'
        if comment:
            filename += '.' + comment
        log_file = os.path.join(save_dir, filename + '.txt')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def train_one_iteration(model, batch_data, optimizer, step, lr_scheduler, scaler, metric_logger,
                            CLIP_GRAD = 5.0):

    input = batch_data['Input'].cuda(non_blocking=True)
    gt = batch_data['Output'].cuda(non_blocking=True)
    with torch.cuda.amp.autocast(enabled=True):
        res = model(input)
        loss = SIC_mse_loss(gt,res)

    with torch.cuda.amp.autocast(enabled=True):        
        with torch.no_grad():
            mae_loss = SIC_mae_loss(gt,res)
    
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    
    if CLIP_GRAD:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
    else:
        grad_norm = get_grad_norm(model.parameters())
    
    scaler.step(optimizer)
    scaler.update()
    lr_scheduler.step_update(step)

    with torch.no_grad():
        metric_logger.update('mse_loss_trace',(loss.cpu(),step))
        metric_logger.update('grad_norm',grad_norm.cpu())
        metric_logger.update('mae_loss_trace',(mae_loss.cpu(),step))

    del grad_norm


    # pass # Test time for iterate dataloader  .__next()__

def validate_training(model,dataloader,val_metric):
    model.eval()
    __dataloader__ = enumerate(dataloader)
    max_iteration = len(dataloader)
    with tqdm(total=max_iteration) as bar:
        with torch.no_grad():
            for i in range(max_iteration):
                _, batch = __dataloader__.__next__()
                # input = batch[0].cuda(non_blocking=True)
                # gt = batch[1].cuda(non_blocking=True)
                input = batch['Input'].cuda(non_blocking=True)
                gt = batch['Output'].cuda(non_blocking=True)
                with torch.cuda.amp.autocast(enabled=True):
                    res = model(input)
                    loss = SIC_mse_loss(gt,res)
                    mae_loss = SIC_mae_loss(gt,res)
                    
                val_metric.update('mse_loss_trace',(loss.cpu(),i))
                val_metric.update('mae_loss_trace',(mae_loss.cpu(),i))

                res = res[0].cpu()
                gt = gt[0].cpu()
                channel_avg_area_diff = []
                channel_avg_iou = []
                for y_hat,y in zip(res,gt):
                    area_diff, iou = SIE_Metric(pred=y_hat,gt=y)  # , _, _
                    channel_avg_area_diff.append(area_diff.item())
                    channel_avg_iou.append(iou.item())
                val_metric.update('sie_regularization_area_trace',(np.mean(channel_avg_area_diff),i))
                val_metric.update('sie_regularization_iou_trace', (np.mean(channel_avg_iou),i))
                bar.update(1)
    model.train()
    torch.cuda.empty_cache()


def save_checkpoint(model, config_setup, ckpt_dir, epoch, iter, record_metrics, optimizer, lr_scheduler, scaler, save_as_latest=True):
    save_states = {'model': model,
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'scaler': scaler.state_dict(),
                  'record_metrics': record_metrics,
                  'epoch': epoch,
                  'iter': iter,
                  'config': config_setup}

    if save_as_latest:
        ckpt_list = os.listdir(ckpt_dir)
        rename = None
        for file in ckpt_list:
            if file.startswith('the_latest_ckpt_epoch'):
                rename = osp.join(ckpt_dir,file.replace('the_latest_',''))
                if osp.exists(rename):
                    print('Overwrite Checkpoint File.')
                    os.remove(rename)
                os.rename(osp.join(ckpt_dir,file), rename)
        save_path = osp.join(ckpt_dir, f'the_latest_ckpt_epoch_{epoch}_{iter}.pth')
    else:
        save_path = osp.join(ckpt_dir, f'ckpt_epoch_{epoch}_{iter}.pth')
    torch.save(save_states, save_path)


def resume_checkpoint(ckpt_dir, ckpt_path=None, ckpt_name=None, load_latest=True):
    if ckpt_dir is None and ckpt_path is not None:
        # Load Designated Ckpt File:
        return torch.load(ckpt_path)

    if osp.exists(ckpt_dir) == False:
        os.makedirs(ckpt_dir)
        return None
    ckpt_list = os.listdir(ckpt_dir)
    if len(ckpt_list) == 0:
        return None
    
    ckpt_file = None
    if load_latest:
        for file in ckpt_list:
            if file.startswith('the_latest_ckpt_epoch'):
                ckpt_file = file
        
        if ckpt_file is None:
            if ckpt_name is None:
                # raise ValueError('Void Latest Checkpoint File Name.')
                print('Void Latest Checkpoint File Name.')
                return None
            else:
                ckpt_file = ckpt_name
    else:
        if ckpt_name is None:
            # raise ValueError('Void Checkpoint File Name.')
            print('Void Checkpoint File Name.')
            return None
        else:
            ckpt_file = ckpt_name

    return torch.load(osp.join(ckpt_dir, ckpt_file))


class MetricLogger(object):
    def __init__(self, name ='Metric', dir='./'):
        self.data = [{}]
        self.name = name
        self.save_dir = dir

        self.data[0]['mse_loss_trace'] = {}
        self.data[0]['mse_loss_trace']['values'] = []
        self.data[0]['mse_loss_trace']['average'] = None
        self.data[0]['mse_loss_trace']['global_steps'] = []

        self.data[0]['mae_loss_trace'] = {}
        self.data[0]['mae_loss_trace']['values'] = []
        self.data[0]['mae_loss_trace']['average'] = None
        self.data[0]['mae_loss_trace']['global_steps'] = []

        self.data[0]['sie_regularization_area_trace'] = {}
        self.data[0]['sie_regularization_area_trace']['values'] = []
        self.data[0]['sie_regularization_area_trace']['average'] = None
        self.data[0]['sie_regularization_area_trace']['global_steps'] = []

        self.data[0]['sie_regularization_iou_trace'] = {}
        self.data[0]['sie_regularization_iou_trace']['values'] = []
        self.data[0]['sie_regularization_iou_trace']['average'] = None
        self.data[0]['sie_regularization_iou_trace']['global_steps'] = []

        self.data[0]['grad_norm'] = None

        self.record_trace_keywords = ['mse_loss_trace', 'mae_loss_trace', 'sie_regularization_area_trace','sie_regularization_iou_trace']
        self.append_keywords = []
        self.average_keywords = ['grad_norm']
    
    def save_log(self, overwrite=True, global_iter='0'):
        if osp.exists(self.save_dir) == False:
            os.makedirs(self.save_dir)

        if overwrite:
            file = osp.join(self.save_dir,self.name + '_Metric_Log')
            if osp.exists(file):
                os.remove(file)
            with open(file, 'wb') as f:
                pkl.dump(self.data, f)
        else:
            timestamp = time.strftime('%m%d_%H_%M_%S') # for Validation
            file = osp.join(self.save_dir,self.name + '_Metric_Log_' + timestamp +'_iter_' + global_iter)
            with open(file, 'wb') as f:
                pkl.dump(self.data, f)

    
    def load_log(self):
        file = osp.join(self.save_dir,self.name + '_Metric_Log')
        if osp.exists(file):
            with open(file, 'rb') as f:
                self.data = []
                self.data.extend(pkl.load(f))
        else:
            raise ValueError('Non-exist Metric Log file.')

    def update(self,k,v):
        if k in self.record_trace_keywords:
            self.data[0][k]['values'].append(v[0])
            if self.data[0][k]['average'] is None:
                self.data[0][k]['average'] = v[0]
            else:
                self.data[0][k]['average'] = (self.data[0][k]['average'] + v[0])/2
            self.data[0]['mse_loss_trace']['global_steps'].append(v[1])

        elif k in self.append_keywords:
            self.data[0][k].append(v)

        elif k in self.average_keywords:
            if self.data[0][k] is None:
                self.data[0][k] = v
            else:
                self.data[0][k] = (self.data[0][k] + v)/2
        else:
            print('Invalid Update!')
            pass

    
    @property
    def get_data(self):
        return self.data[0]



def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm