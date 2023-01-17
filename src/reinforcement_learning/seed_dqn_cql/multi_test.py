import torch
import torch.nn as nn
import os
import numpy as np
import torch.multiprocessing as mp
from multiprocessing import Lock

class Memory(nn.Module):
    def __init__(self):
        super().__init__()
        self.buffer = torch.zeros(10,10)
        self.buffer.share_memory_()
        self.pointer = torch.zeros(1)
        self.pointer.share_memory_()

def add(memory, shared_lock):
    shared_lock.acquire()
    try:
    # import ipdb;ipdb.set_trace()
        memory.buffer += 1
        memory.pointer += 1
        print(f'current pointer is: {memory.pointer}')
    finally:
        shared_lock.release()


if __name__ == '__main__':
    shared_lock = Lock()
    model = Memory()
    model.share_memory()
    processes = []
    for rank in range(20):
        p = mp.Process(target=add, args=(model,shared_lock,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
    print(model.buffer)
