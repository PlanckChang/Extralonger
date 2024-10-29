import numpy as np
import torch
import random
import os
import json

class StandardScaler:
    def __init__(self, mean:float, std:float):
        self.mean = mean
        self.std = std

    def fit_transform(self, data):
        self.mean = data.mean()
        self.std = data.std()
       
        return (data - self.mean) / self.std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):    
        return (data * self.std) + self.mean

def print_log(*values, log=None, end="\n"):
    print(*values, end=end)
    if log:
        if isinstance(log, str):
            log = open(log, "a")
        print(*values, file=log, end=end)
        log.flush()

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  

def set_cpu_num(cpu_num: int):
    os.environ["OMP_NUM_THREADS"] = str(cpu_num)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
    os.environ["MKL_NUM_THREADS"] = str(cpu_num)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
    torch.set_num_threads(cpu_num)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return f"Shape: {obj.shape}"
        elif isinstance(obj, torch.device):
            return str(obj)
        else:
            return super(CustomJSONEncoder, self).default(obj)

def get_index(in_steps=288, stop=16992, out_steps=288):
    start = 0
    stop  = stop+1
    index_new = [
        [i for i in range(start, stop - in_steps - out_steps)],
        [i for i in range(start + in_steps, stop - out_steps)],
        [i for i in range(start + in_steps, stop - out_steps)],
        [i for i in range(start + in_steps + out_steps, stop)]
    ]
    index_new = np.array(index_new).T
    length = len(index_new)
    train_len = int(length * 0.6)
    val_len = int(length * 0.8)
    train = index_new[:train_len]
    val = index_new[train_len:val_len]
    test = index_new[val_len:]
    
    return train, val, test