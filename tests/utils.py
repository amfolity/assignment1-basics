import torch
from torch import Tensor
from jaxtyping import Bool, Float, Int
from typing import Optional
import typing
from collections.abc import Callable, Iterable
from torch import nn
import os
import math

def softmax(x: torch.Tensor, dimension_i : int) -> torch.Tensor:
    exps = torch.exp(x - torch.max(x, keepdim=True, dim=dimension_i)[0])
    summ = torch.sum(exps, dim=dimension_i, keepdim=True)
    return  exps/summ   


def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]):
    #inputs_ = inputs.view(-1, inputs.size(-1))
    #targets_ = targets.view(-1)
    numer = inputs - torch.max(inputs, keepdim=True, dim=-1)[0]
    denumer = torch.sum(torch.exp(numer), dim=-1, keepdim=True)
    return torch.mean(- (torch.gather(numer, dim=-1, index=targets.unsqueeze(-1)) - torch.log(denumer)))
    #return torch.mean(- (numer[torch.arange(numer.size(0)), targets] - torch.log(denumer).squeeze(-1)))


class Adamw(torch.optim.Optimizer):

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):

        defaults = {'lr':lr, 'betas':betas, 'eps':eps, 'weight_decay':weight_decay}
            
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable]=None): # closure: Callable | None
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
                     
            for p in group["params"]:
                
                if p.grad is None:
                    continue
                    
                state = self.state[p]
                
                t = state.get("t", 0) + 1
                
                grad = p.grad.data
                
                exp_avg = state.get('exp_avg', 0.)
                exp_avg_sq = state.get('exp_avg_sq', 0.)
                betas = group['betas']
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                
                exp_avg = exp_avg * betas[0] + (1. - betas[0]) * grad
                exp_avg_sq = exp_avg_sq * betas[1] + (1. - betas[1]) * grad**2
                lr = group["lr"] * math.sqrt(1. - betas[1]**t) / (1. - betas[0]**t)

                p.data -= lr*(exp_avg / (torch.sqrt(exp_avg_sq) + eps))
                p.data -= group["lr"]*weight_decay*p.data
                
                state['exp_avg'] = exp_avg
                state['exp_avg_sq'] = exp_avg_sq
                
                state["t"] = t


def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,):
    
    if it < warmup_iters:
        return (it/warmup_iters) * max_learning_rate
    elif it >= warmup_iters and it <=  cosine_cycle_iters:
        lr = min_learning_rate + 1/2 *(1 + math.cos(((it - warmup_iters)/(cosine_cycle_iters - warmup_iters)) * torch.pi))*(max_learning_rate - min_learning_rate)
        return lr
    else:
        return min_learning_rate


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    eps = 1e-6
    
    norm = math.sqrt(sum([torch.linalg.matrix_norm(parameter.grad, ord=2).item()**2 for parameter in parameters if parameter.grad is not None and len(parameter.shape) >= 2]))
    for parameter in parameters:
        if parameter.grad is None:
            continue
        if norm >= max_l2_norm:
            parameter.grad.data *= max_l2_norm / (norm + eps)


def save_checkpoint(model: torch.nn.Module, optimizer : torch.optim.Optimizer, iteration : int, out : str|os.PathLike|typing.BinaryIO|typing.IO[bytes]):
    model_check = model.state_dict()
    optim_check = optimizer.state_dict()
    dic = {'model': model_check,
          'optimizer': optim_check,
          'iteration': iteration}
    torch.save(dic, out)


def load_checkpoint(src : str|os.PathLike, model, optimizer):
    loaded_dic = torch.load(src)
    model.load_state_dict(loaded_dic["model"])
    optimizer.load_state_dict(loaded_dic["optimizer"])    
    return loaded_dic["iteration"]
    
    
    
    

    
        
    
        







    
        