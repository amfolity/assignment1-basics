import torch.nn as nn
import torch
from einops import rearrange, einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor
from .utils import softmax
import math

class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        with torch.no_grad():
            nn.init.trunc_normal_(self.weight)

    def forward(self, x: torch.Tensor):
        return einsum(x, self.weight, "... d_in, d_model d_in -> ... d_model")


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
        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        summ = (x ** 2).sum(keepdim=True, dim=-1)
        result = (x / torch.sqrt(summ/self.d_model + self.eps)) * self.weight  
        return result.to(in_dtype)


def silu_act(x):
    return x * torch.sigmoid(x)

class Pointwise_Feedforward(nn.Module):

    def __init__(self, d_model, d_ff, activation=silu_act, device=None, dtype=None):
        super().__init__()
        self.activation = silu_act ## storing function
        # self.d_ff = 8 * d_model // 3 
        
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)                       

    def forward(self, x):
        output = self.activation(self.w1(x))
        output = output * (self.w3(x))
        return self.w2(output)

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

    score_logits = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    if mask is not None:
        score_logits[torch.logical_not(mask)] = -float('inf')
    score_logits /= math.sqrt(Q.shape[-1])
    score_logits = softmax(score_logits, -1)
    score_output = einsum(score_logits, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return score_output
  


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(self.d_model, self.d_k * self.num_heads)
        self.k_proj = Linear(self.d_model, self.d_k * self.num_heads)
        self.v_proj = Linear(self.d_model, self.d_k * self.num_heads)
        self.output_proj = Linear(self.d_k * self.num_heads, self.d_model)    

    def forward(self, in_features: Float[Tensor, "... sequence_length d_in"]):

        hidden_shape = (*(in_features.shape[:-1]), self.num_heads, -1)
        
        Q = self.q_proj(in_features)
        Q = rearrange(Q, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads)

        K = self.k_proj(in_features)
        K = rearrange(K, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads)
        
        V = self.v_proj(in_features)
        V = rearrange(V, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads)
        
        mask = torch.ones(*Q.shape[:-1], K.shape[-2], dtype=torch.bool)
        mask = torch.logical_not(torch.triu(mask, diagonal=1))
        
        res = scaled_dot_product_attention(Q, K, V, mask)
        res = rearrange(res, "... num_heads seq_len head_dim -> ... seq_len (num_heads head_dim)")
        res = self.output_proj(res)
        return res


class MultiHeadSelfAttentionWithRope(nn.Module):

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.q_proj = Linear(self.d_model, self.d_k * self.num_heads)
        self.k_proj = Linear(self.d_model, self.d_k * self.num_heads)
        self.v_proj = Linear(self.d_model, self.d_k * self.num_heads)
        self.output_proj = Linear(self.d_k * self.num_heads, self.d_model)        
        
        self.rope = RoPE(self.theta, self.d_k, self.max_seq_len)
        

    def forward(self, in_features: Float[Tensor, "... sequence_length d_in"], 
                      token_positions: Int[Tensor, " ... sequence_length"]):

        hidden_shape = (*(in_features.shape[:-1]), self.num_heads, -1)

        Q = self.q_proj(in_features)
        Q = rearrange(Q, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads)
        Q = self.rope(Q, token_positions)
        
        K = self.k_proj(in_features)
        K = rearrange(K, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads)
        K = self.rope(K, token_positions)
        
        V = self.v_proj(in_features)
        V = rearrange(V, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads)
        
        mask = torch.ones(*Q.shape[:-1], K.shape[-2], dtype=torch.bool)
        mask = torch.logical_not(torch.triu(mask, diagonal=1))
        
        res = scaled_dot_product_attention(Q, K, V, mask)
        res = rearrange(res, "... num_heads seq_len head_dim -> ... seq_len (num_heads head_dim)")
        res = self.output_proj(res)
        return res
    

class TransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        
        self.d_model = d_model 
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        self.ln1 = RMSNorm(d_model, 1e-5)
        self.attn = MultiHeadSelfAttentionWithRope(d_model, num_heads, max_seq_len=None, theta=1e4)
        self.ln2 = RMSNorm(d_model, 1e-5)
        self.ffn = Pointwise_Feedforward(d_model, d_ff, silu_act)

    def forward(self, x: Float[Tensor, "... seq_len d_in"]):

        x_orig = x
        x = self.ln1(x)    
        token_positions = torch.arange(x.shape[-2])[None, :]  
        
        attn1 = self.attn(x, token_positions)          
        ll1_out = x_orig + attn1    
        ll1_out_orig = ll1_out
        
        x2 = self.ln2(ll1_out)
        
        ffn = self.ffn(x2)
        return ffn + ll1_out_orig




        




   