import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler, RandomSampler, BatchSampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data._utils.collate import default_collate

from Learning.utils import worker_init_fn

import os
import os.path as osp
import pickle



class IterationBasedBatchSampler(Sampler):
    """
    Wraps a BatchSampler, resampling from it until a specified number of iterations have been sampled

    References:
        https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration < self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                yield batch
                iteration += 1
                if iteration >= self.num_iterations:
                    break

    def __len__(self):
        return self.num_iterations - self.start_iter


    

class NsidcG02202Data(Dataset):
    def __init__(self,
                 split,
                 dir):
        self.data = []
        print('count pkl data...')
        data_root = osp.join(dir, split)
        file_names = os.listdir(data_root)
        self.size = len(file_names) - 1 # Exclude One Dictionary File
        strs = file_names[0].split('_')
        self.suffix = '_' + strs[1] + '_' + strs[2] + '_' + strs[3]
        self.dir = data_root + '/'
     
    def __getitem__(self, index):
        data = []
        with open(osp.join(self.dir, str(index) + self.suffix), 'rb') as f:
            data.extend(pickle.load(f))
        data[0]['Input'] = torch.tensor(data[0]['Input'])
        data[0]['Output'] = torch.tensor(data[0]['Output'])
        return data[0]
    
    def __len__(self):
        return self.size
    


def build_dataloader(split,task,dir,curr_iteration,epochs,batch_size,num_workers):
    if task == 'IceDiff_FM':
        dataset = NsidcG02202Data(split,dir)
        n_iters_per_epoch = len(dataset) // batch_size
        max_iteration = epochs * n_iters_per_epoch
        
        start_iteration = curr_iteration

        if split == 'TRAIN':
            # shuffle = False
            sampler = RandomSampler(dataset)
            batch_sampler = BatchSampler(sampler,batch_size,drop_last=True)
            batch_sampler = IterationBasedBatchSampler(batch_sampler, max_iteration, start_iteration)
            # drop_last = True
            batch_sz = batch_size

            # sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)

            dataloader = DataLoader(dataset, 
                                # batch_size=batch_sz, 
                                #sampler=sampler,
                                batch_sampler=batch_sampler, # for IterationBasedBatchSampler 
                                # shuffle=shuffle,
                                num_workers=num_workers,
                                pin_memory=True, 
                                # drop_last=drop_last,
                                worker_init_fn=worker_init_fn,
                                collate_fn= collate_fn_train)
        else:
            shuffle = False
            # sampler = SequentialSampler
            drop_last = False
            batch_sz = 1
            
            dataloader = DataLoader(dataset, 
                                batch_size=batch_sz, 
                                #sampler=sampler,
                                # batch_sampler=batch_sampler, 
                                shuffle=shuffle,
                                num_workers=num_workers,
                                pin_memory=True, 
                                drop_last=drop_last,
                                worker_init_fn=worker_init_fn,
                                collate_fn= collate_fn_val)
            

    else:
        dataloader = None
    return dataloader, max_iteration, n_iters_per_epoch


# 8 Weeks
def collate_fn_train(batch):

    batch_num = len(batch)

    ret = {}
    ret['Input'] = torch.zeros((6,8,448,304),dtype=torch.float32)  # B C H W
    ret['Output'] = torch.zeros((6,8,448,304),dtype=torch.float32)
    
    for i in range(batch_num):
        ret['Input'][i] = batch[i]['Input']
        ret['Output'][i] = batch[i]['Output']
    
    return ret


def collate_fn_val(batch):
    batch_num = len(batch)

    if batch_num != 1:
        raise ValueError('Invalid Validation Batch Size!')

    ret = {}
    ret['Input'] = torch.zeros((1,8,448,304),dtype=torch.float32)
    ret['Output'] = torch.zeros((1,8,448,304),dtype=torch.float32)
       
    ret['Input'][0] = batch[0]['Input']
    ret['Output'][0] = batch[0]['Output']
    
    return ret