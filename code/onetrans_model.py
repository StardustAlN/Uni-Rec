"""OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer."""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, NamedTuple, Tuple, Optional, Dict


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict
    seq_lens: dict
    seq_time_buckets: dict


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0), persistent=False)
    
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[:, :seq_len, :].to(device)
        sin = self.sin_cached[:, :seq_len, :].to(device)
        return cos, sin


def apply_rope_to_tensor(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    L = x.shape[2]
    cos_ = cos[:, :L, :].unsqueeze(1)
    sin_ = sin[:, :L, :].unsqueeze(1)
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)
    return x * cos_ + rotate_half(x) * sin_


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class MixedCausalAttention(nn.Module):
    """混合参数化因果注意力：S tokens 共享权重，NS tokens 独立权重"""
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # S tokens 共享的 QKV 投影
        self.q_shared = nn.Linear(d_model, d_model, bias=False)
        self.k_shared = nn.Linear(d_model, d_model, bias=False)
        self.v_shared = nn.Linear(d_model, d_model, bias=False)
        
        # NS tokens 独立的 QKV 投影（每个位置独立）
        self.q_ind = nn.Linear(d_model, d_model, bias=False)
        self.k_ind = nn.Linear(d_model, d_model, bias=False)
        self.v_ind = nn.Linear(d_model, d_model, bias=False)
        
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)
        
    def forward(self, x: torch.Tensor, ns_mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]
        ns_mask: [B, L] bool, True 表示 NS token
        """
        B, L, D = x.shape
        
        # 分离 S 和 NS tokens
        s_mask = ~ns_mask  # [B, L]
        
        # 应用共享投影到所有位置
        q_s = self.q_shared(x)
        k_s = self.k_shared(x)
        v_s = self.v_shared(x)
        
        # 应用独立投影到所有位置
        q_ns = self.q_ind(x)
        k_ns = self.k_ind(x)
        v_ns = self.v_ind(x)
        
        # 根据 mask 混合
        q = torch.where(ns_mask.unsqueeze(-1), q_ns, q_s)
        k = torch.where(ns_mask.unsqueeze(-1), k_ns, k_s)
        v = torch.where(ns_mask.unsqueeze(-1), v_ns, v_s)
        
        # 多头 reshape
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        # RoPE
        cos, sin = self.rope(L, x.device)
        q = apply_rope_to_tensor(q, cos, sin)
        k = apply_rope_to_tensor(k, cos, sin)
        
        # Causal attention mask
        causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        attn_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(attn_mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [B, H, L, D/H]
        
        # 合并输出
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class MixedFFN(nn.Module):
    """混合参数化前馈网络"""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # S tokens 共享
        self.fc1_shared = nn.Linear(d_model, d_ff, bias=False)
        self.fc2_shared = nn.Linear(d_ff, d_model, bias=False)
        
        # NS tokens 独立
        self.fc1_ind = nn.Linear(d_model, d_ff, bias=False)
        self.fc2_ind = nn.Linear(d_ff, d_model, bias=False)
        
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor, ns_mask: torch.Tensor) -> torch.Tensor:
        s_mask = ~ns_mask
        
        # 共享路径
        h_s = self.act(self.fc1_shared(x))
        out_s = self.fc2_shared(h_s)
        
        # 独立路径
        h_ns = self.act(self.fc1_ind(x))
        out_ns = self.fc2_ind(h_ns)
        
        # 混合
        out = torch.where(ns_mask.unsqueeze(-1), out_ns, out_s)
        return out


class OneTransBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int = 2048):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MixedCausalAttention(d_model, n_heads, max_seq_len)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = MixedFFN(d_model, d_ff)
        
    def forward(self, x: torch.Tensor, ns_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), ns_mask)
        x = x + self.ffn(self.ffn_norm(x), ns_mask)
        return x


class OneTransTokenizer:
    """统一 Tokenizer：将所有特征转为 token 序列"""
    def __init__(self, config: dict):
        self.config = config
        self.user_int_dim = config.get('user_int_dim', 46)
        self.item_int_dim = config.get('item_int_dim', 14)
        self.user_dense_dim = config.get('user_dense_dim', 10)
        self.item_dense_dim = config.get('item_dense_dim', 4)
        self.seq_fields = config.get('seq_fields', ['click_seq', 'cart_seq', 'fav_seq', 'buy_seq'])
        
    def tokenize(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        返回:
            tokens: [B, L, D] token 序列
            ns_mask: [B, L] bool, 标记 NS tokens
            seq_info: 序列相关信息
        """
        device = batch['user_int_feats'].device
        B = batch['user_int_feats'].shape[0]
        
        token_list = []
        ns_mask_list = []
        
        # 1. User 静态特征 (S tokens)
        user_int = batch['user_int_feats'].float()  # [B, user_int_dim]
        user_dense = batch['user_dense_feats'].float()  # [B, user_dense_dim]
        user_feat = torch.cat([user_int, user_dense], dim=-1)  # [B, user_int_dim + user_dense_dim]
        token_list.append(user_feat.unsqueeze(1))  # [B, 1, D_user]
        ns_mask_list.append(torch.zeros(B, 1, dtype=torch.bool, device=device))
        
        # 2. Item 特征 (NS tokens)
        item_int = batch['item_int_feats'].float()  # [B, item_int_dim]
        # item_dense 可能为空
        item_dense = batch.get('item_dense_feats', None)
        if item_dense is not None and item_dense.shape[-1] > 0:
            item_dense = item_dense.float()
            item_feat = torch.cat([item_int, item_dense], dim=-1)
        else:
            item_feat = item_int
        token_list.append(item_feat.unsqueeze(1))  # [B, 1, D_item]
        ns_mask_list.append(torch.ones(B, 1, dtype=torch.bool, device=device))
        
        # 3. 序列特征 (混合，带时间戳)
        seq_tokens = []
        seq_ns_masks = []
        
        for field in self.seq_fields:
            if field in batch:
                seq_data = batch[field]  # [B, seq_len] 或 [B, num_slots, seq_len]
                if seq_data.dim() == 3:
                    # 多 slot 序列，取第一个 slot 或拼接
                    seq_data = seq_data[:, 0, :]  # [B, seq_len]
                
                seq_len = seq_data.shape[1]
                
                # 简单的 ID embedding（实际应查表）
                seq_emb = seq_data.float().unsqueeze(-1)  # [B, seq_len, 1]
                # padding 到统一维度
                pad_dim = max(self.user_int_dim + self.user_dense_dim, 
                             self.item_int_dim) - 1
                seq_emb = F.pad(seq_emb, (0, pad_dim))  # [B, seq_len, D]
                
                seq_tokens.append(seq_emb)
                # 序列中有效位置为 NS，padding 为 S
                seq_mask = (seq_data != 0).any(dim=-1) if seq_data.dim() > 2 else (seq_data != 0)
                seq_ns_masks.append(seq_mask)
        
        if seq_tokens:
            # 拼接所有序列
            all_seq = torch.cat(seq_tokens, dim=1)  # [B, total_seq_len, D]
            all_seq_mask = torch.cat(seq_ns_masks, dim=1)  # [B, total_seq_len]
            token_list.append(all_seq)
            ns_mask_list.append(all_seq_mask)
        
        # 拼接所有 tokens
        tokens = torch.cat(token_list, dim=1)  # [B, L, D]
        ns_mask = torch.cat(ns_mask_list, dim=1)  # [B, L]
        
        seq_info = {'seq_lens': {}}
        return tokens, ns_mask, seq_info


class OneTransModel(nn.Module):
    """OneTrans 基础版模型"""
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        d_model = config.get('d_model', 128)
        n_heads = config.get('n_heads', 4)
        d_ff = config.get('d_ff', 512)
        n_layers = config.get('n_layers', 4)
        max_seq_len = config.get('max_seq_len', 512)
        
        # 输入投影
        self.input_proj = nn.Linear(max(56, 18), d_model, bias=False)
        
        # Transformer 栈
        self.layers = nn.ModuleList([
            OneTransBlock(d_model, n_heads, d_ff, max_seq_len)
            for _ in range(n_layers)
        ])
        
        self.final_norm = RMSNorm(d_model)
        
        # 输出头
        self.head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        self.tokenizer = OneTransTokenizer(config)
        
    def forward(self, batch: dict) -> torch.Tensor:
        tokens, ns_mask, seq_info = self.tokenizer.tokenize(batch)
        
        # 投影到 d_model
        x = self.input_proj(tokens)  # [B, L, D]
        
        # Transformer 编码
        for layer in self.layers:
            x = layer(x, ns_mask)
        
        x = self.final_norm(x)
        
        # 提取 item token 的表示（最后一个非序列位置）
        # 简单起见，取第 2 个位置（item 位置）
        item_repr = x[:, 1, :]  # [B, D]
        
        # 预测
        pred = self.head(item_repr).squeeze(-1)  # [B]
        return pred
