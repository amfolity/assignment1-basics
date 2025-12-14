import torch
from torch import Tensor
from jaxtyping import Bool, Float, Int

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
        