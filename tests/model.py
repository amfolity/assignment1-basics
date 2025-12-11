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
        #return x @ self.weight.T
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

    # K = rearrange(K, "... keys d_k -> ... d_k keys")
    score_logits = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    if mask is not None:
        # score_logits += mask
        score_logits[torch.logical_not(mask)] = -float('inf')
    score_logits /= math.sqrt(Q.shape[-1])
    score_logits = softmax(score_logits, -1)
    # score_logits = torch.softmax(score_logits, dim=-1)
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
        self.o_proj = Linear(self.d_k * self.num_heads, self.d_model)
        
        #self.q_proj_weight = nn.Parameter(torch.empty(self.d_k*self.num_heads, self.d_model))
        #self.k_proj_weight = nn.Parameter(torch.empty(self.d_k*self.num_heads, self.d_model))
        #self.v_proj_weight = nn.Parameter(torch.empty(self.d_k*self.num_heads, self.d_model))

        #self.o_proj_weight = nn.Parameter(torch.empty(self.d_model, self.d_k*self.num_heads))
        

    def forward(self, in_features: Float[Tensor, "... sequence_length d_in"]):
        #hidden_dim = (*(in_features)[:-1], self.num_heads, -1)

        hidden_shape = (*(in_features.shape[:-1]), self.num_heads, -1)
        
        #Q = (in_features @ self.q_proj_weight.T).view(hidden_shape).transpose(-2, -3)
        #q_proj_weight = rearrange(self.q_proj_weight, "... d_in d_model -> ... seq_len (num_head head_dim)")
        
        #Q = einsum(in_features, self.q_proj, "... seq_len d_in, ... d_model d_in -> ... seq_len d_model")
        Q = self.q_proj(in_features)

        Q = rearrange(Q, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads)
        
        #K = (in_features @ self.k_proj_weight.T).view(hidden_shape).transpose(-2, -3)

        K = self.k_proj(in_features)
        #K = einsum(in_features, self.k_proj_weight, "... seq_len d_in, ... d_model d_in -> ... seq_len d_model")
        K = rearrange(K, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads)
        
        #V = (in_features @ self.v_proj_weight.T).view(hidden_shape).transpose(-2, -3)

        V = self.v_proj(in_features)
        #V = einsum(in_features, self.v_proj_weight, "... seq_len d_in, ... d_model d_in -> ... seq_len d_model")
        V = rearrange(V, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads)
        
        mask = torch.ones(*Q.shape[:-1], K.shape[-2], dtype=torch.bool)
        mask = torch.logical_not(torch.triu(mask, diagonal=1))
        res = scaled_dot_product_attention(Q, K, V, mask)
        output_shape = ((*in_features.shape[:-1], -1))
        
        #res = res.transpose(-2,-3).reshape(output_shape).contiguous()
        #return res @ self.o_proj_weight.T
        res = rearrange(res, "... num_heads seq_len head_dim -> ... seq_len (num_heads head_dim)")
        res = self.o_proj(in_features)
        #res = einsum(res, self.o_proj, "... seq_len d_in, ... d_model d_in -> ... seq_len d_model")
        return res


class MultiHeadSelfAttentionWithRope(nn.Module):

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta

        #self.q_proj_weight = nn.Parameter(torch.empty(self.d_k*self.num_heads, self.d_model))
        #self.k_proj_weight = nn.Parameter(torch.empty(self.d_k*self.num_heads, self.d_model))
        #self.v_proj_weight = nn.Parameter(torch.empty(self.d_k*self.num_heads, self.d_model))

        #self.o_proj_weight = nn.Parameter(torch.empty(self.d_model, self.d_k*self.num_heads))

        self.q_proj = Linear(self.d_model, self.d_k * self.num_heads)
        self.k_proj = Linear(self.d_model, self.d_k * self.num_heads)
        self.v_proj = Linear(self.d_model, self.d_k * self.num_heads)
        self.o_proj = Linear(self.d_k * self.num_heads, self.d_model)        
        
        self.rope = RoPE(self.theta, self.d_k, self.max_seq_len)
        

    def forward(self, in_features: Float[Tensor, "... sequence_length d_in"], 
                    token_positions: Int[Tensor, " ... sequence_length"]):
        #hidden_dim = (*(in_features)[:-1], self.num_heads, -1)

        hidden_shape = (*(in_features.shape[:-1]), self.num_heads, -1)
        #Q = (in_features @ self.q_proj_weight.T).view(hidden_shape).transpose(-2, -3)
        #q_proj_weight = rearrange(self.q_proj_weight, "... d_in d_model -> ... seq_len (num_head head_dim)")
        #Q = einsum(in_features, self.q_proj, "... seq_len d_in, ... d_model d_in -> ... seq_len d_model")

        Q = self.q_proj(in_features)
        Q = rearrange(Q, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads)
        Q = self.rope(Q, token_positions)
        
        #K = (in_features @ self.k_proj_weight.T).view(hidden_shape).transpose(-2, -3)
        
        #K = einsum(in_features, self.k_proj, "... seq_len d_in, ... d_model d_in -> ... seq_len d_model")
        K = self.k_proj(in_features)
        K = rearrange(K, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads)
        K = self.rope(K, token_positions)
        
        #V = (in_features @ self.v_proj_weight.T).view(hidden_shape).transpose(-2, -3)

        #V = einsum(in_features, self.v_proj, "... seq_len d_in, ... d_model d_in -> ... seq_len d_model")
        V = self.v_proj(in_features)
        V = rearrange(V, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads)
        V = self.rope(V, token_positions)
        
        mask = torch.ones(*Q.shape[:-1], K.shape[-2], dtype=torch.bool)
        mask = torch.logical_not(torch.triu(mask, diagonal=1))
        res = scaled_dot_product_attention(Q, K, V, mask)
        output_shape = ((*in_features.shape[:-1], -1))
        
        #res = res.transpose(-2,-3).reshape(output_shape).contiguous()
        #return res @ self.o_proj_weight.T
        res = rearrange(res, "... num_heads seq_len head_dim -> ... seq_len (num_heads head_dim)")
        res = self.o_proj(in_features)
        #res = einsum(res, self.o_proj, "... seq_len d_in, ... d_model d_in -> ... seq_len d_model")
        return res
    

class transformer_block(nn.Module):

    def __init__(self, d_model, num_heads, d_ff):
        self.d_model = d_model 
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        self.ln1 = RMSNorm(d_model, 1e-5)
        self.attention = MultiHeadSelfAttentionWithRope(d_model, num_heads, max_seq_len=None, theta=1e4)
        self.ln2 = RMSNorm(d_model, 1e-5)
        self.ffn = Pointwise_Feedforward(d_model, d_ff, silu_act)

    def forward(self, x: Float[Tensor, "... seq_len d_in"]):

        x_orig = x.clone()
        x = self.ln1(x)    
        token_positions = torch.arange(in_features.shape[-2])[None, :]  
        
        attn1 = self.attention(x, token_positions)          
        ll1_out = in_features_orig + attn1    
        ll1_out_orig = ll1_out.clone()
        
        x2 = self.ln2(ll1_out)
        
        ffn = self.ffn(x2)
        return ffn + ll1_out_orig




        




   