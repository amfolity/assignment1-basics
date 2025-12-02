import torch.nn as nn
import torch
from einops import rearrange, einsum
from .utils import softmax

class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.w = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        with torch.no_grad():
            nn.init.trunc_normal_(self.w)

    def forward(self, x: torch.Tensor):
        return x @ self.w.T


class Embedding(nn.Module):

    def __init__(self, num_embedding, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.embed = nn.Parameter(torch.empty(num_embedding, embedding_dim, device=device, dtype=dtype))
        with torch.no_grad():
            nn.init.trunc_normal_(self.embed)
        

    def forward(self, token_ids: torch.Tensor):
        return self.embed[token_ids]


class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype = None):
        super().__init__()
        self.w = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        summ = (x ** 2).sum(keepdim=True, dim=-1)
        result = (x / torch.sqrt(summ/self.d_model + self.eps)) * self.w     
        return result.to(in_dtype)


def silu_act(x):
    return x * torch.sigmoid(x)

class Pointwise_Feedforward(nn.Module):

    def __init__(self, d_model, d_ff, activation=silu_act, device=None, dtype=None):
        super().__init__()
        self.activation = silu_act ## storing function
        # self.d_ff = 8 * d_model // 3 
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
                               

    def forward(self, x):
        output = self.activation(x @ self.w1.T)
        output = output * (x @ self.w3.T)
        output = output @ self.w2.T
        return output


class RoPE(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.device = device
        self.d_k = d_k
        
        Theta = (theta**( - torch.arange(0, self.d_k, 2) / self.d_k )).to(device=device, dtype=torch.float32)
        Theta = Theta.repeat_interleave(2, dim=-1)
        
        self.register_buffer("Theta", Theta, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:

        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        xx = torch.empty_like(x) 
        xx[..., ::2] = -x2
        xx[..., 1::2] = x1
        coeff = token_positions[..., None].to(dtype=self.Theta.dtype) @ self.Theta[None, :]  ## m Theta --> position by shift
        return x*torch.cos(coeff) + xx*torch.sin(coeff)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None) -> Float[Tensor, " ... queries d_v"]:

    K = rearrange(K, "... keys d_k -> ... d_k keys")
    score = einsum(Q, K, "... queries d_k, ... d_k keys -> ... queries keys")
    if mask is not None:
        score_logits += mask
    score_logits /= math.sqrt(Q.shape[-1])
    score_logits = softmax(score_logits, -1)
    score_output = einsum(score_logits, V, "... queries , ... values d_v -> ... queries d_v")
    return score_output
    
    #d_model: int,
    #num_heads: int,
    #q_proj_weight: Float[Tensor, " d_k d_in"],
    #k_proj_weight: Float[Tensor, " d_k d_in"],
    #v_proj_weight: Float[Tensor, " d_v d_in"],
    #o_proj_weight: Float[Tensor, " d_model d_v"],
    #in_features: Float[Tensor, " ... sequence_length d_in"]):

    #hidden_dim = (*(in_features)[:-1], num_heads, -1)
    #q_proj_weight = rearrange(q_proj_weight.view(hidden_dim), "... sequence_length num_head head_size(d_k) -> ... num_head sequence_length head_size(d_k)")
        



        















        












        