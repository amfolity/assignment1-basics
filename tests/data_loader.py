from typing import List
import numpy as np
import torch
import random


def data_loading(x, batch_size, context_length, device) -> tuple[torch.Tensor, torch.Tensor]: ## change List to NPARRAY

    # working version
    inputs = []
    outputs = []

    for b in range(batch_size):
        idx = random.randint(0, x.shape[0] - context_length - 1)
        inputs.append(x[idx: idx + context_length])
        outputs.append(x[idx+1: idx + context_length+1])
        
    final_input = torch.from_numpy(np.vstack(inputs)).to(device)
    final_output = torch.from_numpy(np.vstack(outputs)).to(device)
    
    return final_input, final_output
    #######

    # random start indices
    #idx = np.random.randint(0, x - context_length - 1, size=batch_size)
    
    # offsets for one batch
    #offsets = np.arange(idx, batch_size)
    
    # build 2D index matrix: shape (num_of_batches, batch_size)
    #arr_idx = idx[:, None] + offsets[None, :]
    
    # gather and flatten
    #final_batch = x[arr_idx].reshape(-1, x.shape[1:])  # if x is multidimensional

    #x = torch.as_tensor(x, device=device)

    #starts = torch.randint(
    #    0,
    #    x.shape[0] - context_length - 1,
    #    (batch_size,),
    #    device=device
    #)
    
    #idx = starts[:, None] + torch.arange(context_length-1, device=device)
    
    #final_input  = x[idx]#.reshape(-1)
    #final_output = x[idx + 1]#.reshape(-1)
    #final_input  = inputs.reshape(batch_size * context_length, *x.shape[1:])
    #final_output = outputs.reshape(batch_size * context_length, *x.shape[1:])
    #return final_input, final_output


    
