import argparse
import yaml
import torch
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
# from sampler_core import ParallelSampler, TemporalGraphBlock

root_dir = '/root/swift'
if root_dir not in sys.path:
    sys.path.append(root_dir)
# from sampler.sampler_core import ParallelSampler, TemporalGraphBlock
from utils import *

class NegLinkSampler:

    def __init__(self, num_nodes, num_edges):
        self.num_nodes = num_nodes
        self.test_nodes = None
        self.num_edges = num_edges

    def sample(self, n):
        return np.random.randint(self.num_nodes, size=n)
    
    def load_test(self):
        if (not os.path.exists(f'/root/swift/test_set/{self.num_nodes}.bin')):
            res = np.random.randint(self.num_nodes, size=self.num_edges, dtype = np.int32)
            res.tofile(f'/root/swift/test_set/{self.num_nodes}.bin')
        else:
            res = np.fromfile(f'/root/swift/test_set/{self.num_nodes}.bin', dtype = np.int32)
        
        self.test_nodes = res
    def sample_test(self, left, right):
        if (self.test_nodes is None):
            self.load_test()
        return self.test_nodes[left:right]
    

class ReNegLinkSampler:

    pre_res = None
    pre_ttl = None
    ratio = 1

    def __init__(self, num_nodes, ratio):
        self.ratio = ratio
        self.pre_res = None
        self.num_nodes = num_nodes

    def sample(self, n):

        if (self.ratio <= 0 or (self.pre_res is not None and n < self.pre_res.shape[0])):
            return np.random.randint(self.num_nodes, size=n)
        
        cur_res = np.zeros(n, dtype = np.int32)
        cur_ttl = np.zeros(n, dtype = np.int32)
        if (self.pre_res is None):
            cur_res[:] = np.random.randint(self.num_nodes, size=n)
        else:
            reuse_num = int(self.pre_res.shape[0] * self.ratio)
            pre_ttl_ind = np.argsort(self.pre_ttl)
            pre_reuse_ind = pre_ttl_ind[:reuse_num]
            cur_res[:reuse_num] = self.pre_res[pre_reuse_ind]
            cur_ttl[:reuse_num] = self.pre_ttl[pre_reuse_ind] + 1
            cur_res[reuse_num:] = np.random.randint(self.num_nodes, size=cur_res.shape[0] - reuse_num)
            
            ind = torch.randperm(cur_res.shape[0])
            cur_res = cur_res[ind]
            cur_ttl = cur_ttl[ind]
            
        sameNum = 0
        if (self.pre_res is not None):
            sameNum = np.sum(np.isin(cur_res, self.pre_res))
        self.pre_res = cur_res
        self.pre_ttl = cur_ttl

        return cur_res
    

import math
from config.train_conf import *
class TrainNegLinkSampler:

    def __init__(self, num_nodes, num_edges, k = 8):

        config = GlobalConfig()
        block_size = config.pre_sample_size
        block_num = math.ceil(num_edges/ block_size)

        block_neg = torch.randint(0, num_nodes, (num_edges,))

        self.part = []
        nodes = torch.randperm(num_nodes, dtype = torch.int32)
        
        per_node_num = block_size
        left, right = 0, 0
        self.k = k
        
        for i in range(block_num):
            self.part.append(block_neg[i * block_size: min((i+1) * block_size, block_neg.shape[0])])

        self.num_nodes = num_nodes

        self.ptr = 0

        

    def sample(self, n, i = 0, cur_batch = 0):
        part_len = len(self.part)
        i = self.ptr % part_len
        res = np.random.choice(self.part[i], size=n, replace = True)
        self.ptr += 1
        return res

class NegLinkInductiveSampler:
    def __init__(self, nodes):
        self.nodes = list(nodes)

    def sample(self, n):
        return np.random.choice(self.nodes, size=n)
    

    