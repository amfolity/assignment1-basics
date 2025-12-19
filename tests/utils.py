import torch
from torch import Tensor
from jaxtyping import Bool, Float, Int
from typing import Optional
from collections.abc import Callable
from torch import nn
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

    def __init__(self, params, lr, betas, eps, weight_decay):

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
        







    
        