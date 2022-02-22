# Modified by Peize Sun, sunpeize@foxmail.com
# From https://github.com/SysCV/qdtrack/blob/master/qdtrack/datasets/samplers/distributed_video_sampler.py
# 
import os
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


class DistributedVideoSampler(Sampler):

    def __init__(self, dataset, start_id=0, num_replicas=None, rank=None, local_rank=None, local_size=None, shuffle=False):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        
        self.shuffle = shuffle
        assert not self.shuffle, 'Specific for video sequential testing.'
        
        first_frame_indices = []
        
        imgs = self.dataset.coco.imgs
        for i, id_in_dataset in enumerate(self.dataset.ids):
            img_info  = imgs[id_in_dataset]
            if img_info['frame_id'] == start_id:
                first_frame_indices.append(i)

        chunks = np.array_split(first_frame_indices, num_replicas)
        
        split_flags = [c[0] for c in chunks]
        split_flags.append(len(dataset))
                
        self.indices = [
            list(range(split_flags[i], split_flags[i + 1]))
            for i in range(self.num_replicas)
        ]
        self.num_samples = max([len(indice) for indice in self.indices])
        
    def __iter__(self):
        indices = self.indices[self.rank]
        return iter(indices)
    
    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

