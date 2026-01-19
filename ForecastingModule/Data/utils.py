import numpy as np
from torch.utils.data.sampler import Sampler
import copy

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
    
class MovingAverage(object):
    '''
    For Generating Multi-Granularity Datasets Based on Daily Data.
    Slide Step is set default to 1.
    '''
    def __init__(self, shape=(448,304), num_average_window=12, period=30):
        self.num_average_window = num_average_window
        self.period = period

        # Allocation of Storage Space
        self.window_tensor = np.zeros((num_average_window,period,shape[0],shape[1]),dtype=np.float32)
        self.window_averages = np.zeros((num_average_window,shape[0],shape[1]),dtype=np.float32)
        
        self.iter = None

    def insert_daily_data(self,ndarray,window_index,period_index):
        self.window_tensor[window_index][period_index] = ndarray

    @property
    def average(self):
        for i in range(self.num_average_window):
            self.window_averages[i] = np.mean(self.window_tensor[i],axis=0)
        return self.window_averages
    
    def init_moving_average_iter(self):
        self.iter = 0
    
    def slide(self,ndarray):
        update = copy.deepcopy(ndarray)
        # Moving
        for i in range(self.num_average_window-1,-1,-1):
            pop = copy.deepcopy(self.window_tensor[i][self.iter])
            self.window_tensor[i][self.iter] = update
            update = copy.deepcopy(pop)

        self.iter += 1
        if self.iter >= self.period:
            self.iter = 0 # Cycle Moving
        return update