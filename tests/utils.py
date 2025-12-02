import torch

def softmax(x: torch.Tensor, dimension_i : int) -> torch.Tensor:
    exps = torch.exp(x - torch.max(x, keepdim=True, dim=dimension_i)[0])
    summ = torch.sum(exps, dim=dimension_i, keepdim=True)
    return  exps/summ