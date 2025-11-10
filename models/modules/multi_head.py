"""
    Multi-head attention module.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import attention
from . import utils

from torch import Tensor
from typing import Optional, Union, Tuple, List
from mamba_ssm import Mamba
from torch_geometric.nn.inits import glorot


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_q: int = None,
        d_k: int = None,
        d_v: int = None,
        d_model: int = None,
        n_head: int = 1,
        qkv_bias: bool = False,
        attn_drop: float = 0,
        num_edge_type: int = 2,
    ) -> None:
        super().__init__()
        self.num_heads = n_head
        self.hidden_dims = d_model
        self.attention = attention.MultiRelationalSelfAttention(num_heads=n_head, dropout=attn_drop, num_edge_type=num_edge_type)
        assert d_q is not None and d_k is not None and d_v is not None and d_model is not None, "Please specify the dimensions of Q, K, V and d_model"
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        # q: q_dims, k: k_dims, v: v_dims, d: hidden_dims, h: num_heads, d_i: dims of each head
        self.W_q = nn.Linear(d_q, d_model, bias=qkv_bias)  # (q, h*d_i=d)
        self.W_k = nn.Linear(d_k, d_model, bias=qkv_bias)  # (k, h*d_i=d)
        self.W_v = nn.Linear(d_v, d_model, bias=qkv_bias)  # (v, h*d_i=d)
        self.W_o = nn.Linear(d_model, d_model, bias=qkv_bias)  # (h*d_i=d, d)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor = None,
        adjacency_matrix: Union[Tensor, List[Tensor]] = None,
    ):
        # b: batch_size, h:num_heads, l: seq_len, d: d_hidden
        b, h, l, d_i = queries.shape[0], self.num_heads, queries.shape[1], self.hidden_dims // self.num_heads
        Q, K, V = self.W_q(queries), self.W_k(keys), self.W_v(values)  # (b, l, h*d_i=d)
        Q, K, V = [M.view(b, l, h, d_i).permute(0, 2, 1, 3) for M in (Q, K, V)]  # (b, h, l, d_i)
        attn_out = self.attention(Q, K, V, attention_mask, adjacency_matrix)
        out, attn_weight = attn_out["out"], attn_out["attn_weight"]
        out = out.permute(0, 2, 1, 3).contiguous().view(b, l, h * d_i)  # (b, l, h*d_i=d)

        return {
            "out": self.W_o(out),  # (b, l, d)
            "attn_weight": attn_weight,  # (b, l, l) | (b, h, l, l)
        }

class GraphMamba(nn.Module):
    def __init__(self, d_model, num_edge_type):
        super().__init__()
        self.d_model = d_model
        self.num_edge_type = num_edge_type
        self.attention = Mamba(
            d_model = d_model,
            d_state = 32,
            d_conv = 4,
            expand = 1,
        )
        self.weight_E = nn.ParameterDict()
        for typ in range(num_edge_type):
            self.weight_E[str(typ)] = nn.Parameter(torch.empty(1, 1, 1, 1), requires_grad=True)

        if num_edge_type > 0:
            self.reset_parameters()

    def reset_parameters(self):
        for typ in range(self.num_edge_type):
            glorot(self.weight_E[str(typ)])

    def forward(self, x, attention_mask, adjacency_matrix):
        M, A = attention_mask, adjacency_matrix
        assert type(A) == list
        A = [_A.unsqueeze(1) for _A in A]

        if attention_mask is not None:
            attention_mask = attention_mask.to(x.dtype)
            x = x * attention_mask.unsqueeze(-1)

        h_attn = self.attention(x)

        if attention_mask is not None:
            attention_mask = attention_mask.to(h_attn.dtype)
            attn_score = h_attn * attention_mask.unsqueeze(-1)  # broadcast to [B, L, D]

        if self.num_edge_type > 0:
            for typ in range(self.num_edge_type):
                edge_weighted = A[typ] * self.weight_E[str(typ)]   # (B, 1, L, L)
                update = torch.einsum("bij,bjd->bid", edge_weighted.squeeze(1), attn_score)
                B_E = update if typ == 0 else B_E + update
        else:
            B_E = None

        attn_score = attn_score + B_E if B_E is not None else attn_score
        return attn_score

class GraphMamba1(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = Mamba(
            d_model = d_model,
            d_state = 128,
            d_conv = 4,
            expand = 2,
        )

    def forward(self, x, attention_mask):
        M = attention_mask

        if M is not None:
            M = M.to(x.dtype)
            x = x * M.unsqueeze(-1)

        h_attn = self.attention(x)

        if M is not None:
            M = M.to(h_attn.dtype)
            attn_score = h_attn * M.unsqueeze(-1)  # broadcast to [B, L, D]

        return attn_score

class GraphMamba2(nn.Module):
    def __init__(self, num_edge_type):
        super().__init__()
        self.num_edge_type = num_edge_type
        
        self.weight_E = nn.ParameterDict()
        for typ in range(num_edge_type):
            self.weight_E[str(typ)] = nn.Parameter(torch.empty(1, 1, 1, 1), requires_grad=True)

        if num_edge_type > 0:
            self.reset_parameters()

    def reset_parameters(self):
        for typ in range(self.num_edge_type):
            glorot(self.weight_E[str(typ)])

    def forward(self, x, attention_mask, adjacency_matrix):
        M, A = attention_mask, adjacency_matrix
        assert type(A) == list
        A = [_A.unsqueeze(1) for _A in A]

        if attention_mask is not None:
            attention_mask = attention_mask.to(x.dtype)
            x = x * attention_mask.unsqueeze(-1)

        h_attn = x

        if attention_mask is not None:
            attention_mask = attention_mask.to(h_attn.dtype)
            attn_score = h_attn * attention_mask.unsqueeze(-1)  # broadcast to [B, L, D]

        if self.num_edge_type > 0:
            for typ in range(self.num_edge_type):
                edge_weighted = A[typ] * self.weight_E[str(typ)]   # (B, 1, L, L)
                update = torch.einsum("bij,bjd->bid", edge_weighted.squeeze(1), attn_score)
                B_E = update if typ == 0 else B_E + update
        else:
            B_E = None

        attn_score = attn_score + B_E if B_E is not None else attn_score
        return attn_score



if __name__ == "__main__":
    # Test multi-head attention
    # b, l, d, h = 2, 4, 4, 2
    # q, k, v = torch.randn(b, l, d), torch.randn(b, l, d), torch.randn(b, l, d)
    # mask = utils.valid_length_to_mask(torch.tensor([2, 3]), max_len=l)
    # attn = MultiHeadAttention(d_q=d, d_k=d, d_v=d, d_model=d, n_head=h, attn_type="SelfAttention", qkv_bias=True)
    # out = attn(q, k, v, mask)
    # print(out.shape)
    x = torch.tensor([
        [  # 第一个样本（B=0）
            [1.0, 1.1, 1.2, 1.3],   # L=0, 有效
            [2.0, 2.1, 2.2, 2.3],   # L=1, 有效
            [9.9, 9.9, 9.9, 9.9],   # L=2, padding（非零）
        ],
        [  # 第二个样本（B=1）
            [3.0, 3.1, 3.2, 3.3],   # L=0, 有效
            [8.8, 8.8, 8.8, 8.8],   # L=1, padding
            [8.8, 8.8, 8.8, 8.8],   # L=2, padding
        ]
    ])
    mask = torch.tensor([
        [1, 1, 0],  # 第一个样本，前两步有效
        [1, 0, 0],  # 第二个样本，只有第一步有效
    ])
    A = [torch.tensor([
        [0.49, 0.18, 0.30, 0.03],
        [0.18, 0.49, 0.30, 0.03],
        [0.26, 0.26, 0.35, 0.13],
        [0.03, 0.06, 0.20, 0.71],
    ])]  # shape: (4, 2)
    attn=GraphMamba(d_model=512, num_edge_type=1)
    out=attn(x=x,attention_mask=mask,adjacency_matrix=A)
    print(out)




