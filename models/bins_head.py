import math
import numbers
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Simple MLP used by the bin-width head and mask embedding head."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        assert num_layers >= 1
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x, inplace=True)
        return x


class SinePositionalEncoding2D(nn.Module):
    """2D sine-cosine positional encoding for feature maps converted to tokens."""

    def __init__(self, num_feats: int, normalize: bool = True, scale: float = 2 * math.pi, cache_limit: int = 16):
        super().__init__()
        self.num_feats = int(num_feats)
        self.normalize = bool(normalize)
        self.scale = float(scale)
        self.cache_limit = int(cache_limit)
        self._cache = {}

    def clear_cache(self):
        self._cache.clear()

    def _build_pos_base(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        y_embed = torch.arange(h, device=device).unsqueeze(1).repeat(1, w)
        x_embed = torch.arange(w, device=device).unsqueeze(0).repeat(h, 1)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (h - 1 + eps) * self.scale
            x_embed = x_embed / (w - 1 + eps) * self.scale
        dim_t = torch.arange(self.num_feats, device=device, dtype=torch.float32)
        dim_t = 10000 ** (2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        pos = torch.cat((pos_y, pos_x), dim=-1)
        return pos.view(h * w, -1).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        key = (int(h), int(w), str(x.device))
        pos_base = self._cache.get(key, None)
        if pos_base is None or pos_base.device != x.device:
            pos_base = self._build_pos_base(h, w, x.device)
            self._cache[key] = pos_base
            if len(self._cache) > self.cache_limit:
                self._cache.pop(next(iter(self._cache)))
        return pos_base.repeat(b, 1, 1).to(dtype=x.dtype)


class BinsFormerDetrBlock(nn.Module):
    """Single DETR-style decoder block used by the BinsFormer ablation head."""

    def __init__(self, dim: int, num_heads: int = 8, ffn_dim: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.ReLU(inplace=True), nn.Linear(ffn_dim, dim))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, q: torch.Tensor, q_pos: torch.Tensor, mem: torch.Tensor, mem_pos: torch.Tensor) -> torch.Tensor:
        q_with_pos = q + q_pos
        q2, _ = self.self_attn(q_with_pos, q_with_pos, q, need_weights=False)
        q = self.norm1(q + self.drop(q2))

        q_with_pos = q + q_pos
        k = mem + mem_pos
        q2, _ = self.cross_attn(q_with_pos, k, mem, need_weights=False)
        q = self.norm2(q + self.drop(q2))

        q2 = self.ffn(q)
        q = self.norm3(q + self.drop(q2))
        return q


class BinsFormerDetrDecoder(nn.Module):
    """Multi-level DETR decoder used to reproduce the BinsFormer-style ablation head."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        blocks_per_level: Sequence[int],
        operation: str = "//",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.operation = str(operation)
        self.blocks_per_level = [int(x) for x in blocks_per_level]
        self.num_levels = len(self.blocks_per_level)
        total_blocks = sum(self.blocks_per_level)
        if total_blocks <= 0:
            raise ValueError('blocks_per_level must sum to > 0')

        self.blocks = nn.ModuleList([
            BinsFormerDetrBlock(dim, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout)
            for _ in range(total_blocks)
        ])

        self.block2level = []
        if self.operation == "//":
            for lv, nb in enumerate(self.blocks_per_level):
                self.block2level.extend([lv] * nb)
        elif self.operation == "%":
            remain = self.blocks_per_level[:]
            lv = 0
            while sum(remain) > 0:
                if remain[lv] > 0:
                    self.block2level.append(lv)
                    remain[lv] -= 1
                lv = (lv + 1) % self.num_levels
            if len(self.block2level) != total_blocks:
                raise RuntimeError("Internal schedule error for operation='%'")
        else:
            raise ValueError(f"Unsupported operation={operation}, use '//' or '%'")

    def forward(self, q: torch.Tensor, q_pos: torch.Tensor, mem_list: List[torch.Tensor], pos_list: List[torch.Tensor]) -> torch.Tensor:
        if len(mem_list) != self.num_levels or len(pos_list) != self.num_levels:
            raise ValueError(f'mem_list/pos_list must have {self.num_levels} levels, got {len(mem_list)}/{len(pos_list)}')
        for blk, lv in zip(self.blocks, self.block2level):
            q = blk(q, q_pos, mem_list[lv], pos_list[lv])
        return q


def to_3d(x):
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)


def to_4d(x, h, w):
    b, hw, c = x.shape
    assert hw == h * w
    return x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        self.body = BiasFree_LayerNorm(dim) if LayerNorm_type == "BiasFree" else WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, _, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        head = self.num_heads
        _, c, _, _ = q.shape
        head_dim = c // head

        q = q.view(b, head, head_dim, h, w).reshape(b, head, head_dim, h * w)
        k = k.view(b, head, head_dim, h, w).reshape(b, head, head_dim, h * w)
        v = v.view(b, head, head_dim, h, w).reshape(b, head, head_dim, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.contiguous().view(b, head, head_dim, h, w).reshape(b, head * head_dim, h, w)
        return self.project_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type="WithBias"):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class FEBR(nn.Module):
    """Feature-Enhanced Bin Refinement.

    This module keeps the pooling-based readout path used by the final TSONet,
    including single-stage pooling, multi-stage late fusion, and the final
    iterative coarse-to-fine query readout.
    """

    def __init__(
        self,
        in_channels_by_level: Dict[int, int],
        fuse_levels: Sequence[int],
        hidden_dim: int,
        n_bins: int,
        refine_blocks: Sequence[int] = (1, 1, 1),
        heads: int = 8,
        ffn_expansion_factor: float = 2.0,
        bias: bool = False,
        LayerNorm_type: str = "WithBias",
        attn_temp: float = 1.0,
        align_corners: bool = True,
        pool_mode: str = "single",
        pool_fuse_learnable: bool = True,
        pool_iter_anchor_mix: bool = False,
        pool_iter_anchor_mix_mode: str = "per_stage",
        pool_iter_anchor_mix_init: float = -2.0,
        pool_iter_tau_mode: str = "shared",
        pool_iter_tau_init: float = 1.0,
        pool_iter_tau_clamp: Tuple[float, float] = (0.1, 10.0),
    ):
        super().__init__()
        self.fuse_levels = [int(x) for x in fuse_levels]
        if len(self.fuse_levels) < 1:
            raise ValueError("fuse_levels must be non-empty")

        self.hidden_dim = int(hidden_dim)
        self.n_bins = int(n_bins)
        self.align_corners = bool(align_corners)
        self.pool_mode = str(pool_mode)
        self.pool_fuse_learnable = bool(pool_fuse_learnable)

        self.proj = nn.ModuleDict()
        for lv in self.fuse_levels:
            if lv not in in_channels_by_level:
                raise ValueError(f"in_channels_by_level missing level={lv}")
            self.proj[str(lv)] = nn.Conv2d(int(in_channels_by_level[lv]), self.hidden_dim, kernel_size=1, bias=False)

        if isinstance(refine_blocks, str):
            refine_blocks = [int(x) for x in refine_blocks.split(',') if x.strip()]
        elif isinstance(refine_blocks, int):
            refine_blocks = [int(refine_blocks)] * len(self.fuse_levels)
        refine_blocks = [int(x) for x in refine_blocks]
        if len(refine_blocks) != len(self.fuse_levels):
            raise ValueError("refine_blocks length must match fuse_levels length")

        self.refine = nn.ModuleList()
        for nb in refine_blocks:
            self.refine.append(nn.Sequential(*[
                TransformerBlock(
                    dim=self.hidden_dim,
                    num_heads=heads,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(nb)
            ]))

        self.attn_temp = nn.Parameter(torch.tensor(float(attn_temp)))
        self.out_norm = nn.LayerNorm(self.hidden_dim)

        self.pool_stage_w = None
        if self.pool_mode == "ms_late" and self.pool_fuse_learnable:
            self.pool_stage_w = nn.Parameter(torch.zeros(len(self.fuse_levels), dtype=torch.float32))

        self.pool_iter_anchor_mix = bool(pool_iter_anchor_mix)
        self.pool_iter_anchor_mix_mode = str(pool_iter_anchor_mix_mode)
        self.pool_iter_tau_mode = str(pool_iter_tau_mode)
        self.pool_iter_tau_clamp = tuple(pool_iter_tau_clamp)

        self.pool_iter_tau = None
        if self.pool_iter_tau_mode == "per_stage":
            self.pool_iter_tau = nn.Parameter(
                torch.full((len(self.fuse_levels),), float(pool_iter_tau_init), dtype=torch.float32)
            )

        self.pool_iter_mix = None
        if self.pool_iter_anchor_mix:
            if self.pool_iter_anchor_mix_mode == "scalar":
                self.pool_iter_mix = nn.Parameter(torch.tensor(float(pool_iter_anchor_mix_init), dtype=torch.float32))
            elif self.pool_iter_anchor_mix_mode == "per_stage":
                self.pool_iter_mix = nn.Parameter(
                    torch.full((len(self.fuse_levels),), float(pool_iter_anchor_mix_init), dtype=torch.float32)
                )
            else:
                raise ValueError(
                    f"pool_iter_anchor_mix_mode must be 'scalar' or 'per_stage', got {self.pool_iter_anchor_mix_mode}"
                )

    def _pool_stage_weights(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        s = len(self.fuse_levels)
        if self.pool_stage_w is None:
            return torch.ones(s, device=device, dtype=dtype) / float(s)
        return torch.softmax(self.pool_stage_w.to(device=device, dtype=dtype), dim=0)

    def _get_iter_tau(self, stage_idx: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        lo, hi = self.pool_iter_tau_clamp
        if self.pool_iter_tau_mode == "per_stage" and self.pool_iter_tau is not None:
            tau = self.pool_iter_tau.to(device=device, dtype=dtype)[stage_idx]
        else:
            tau = self.attn_temp.to(device=device, dtype=dtype)
        return tau.clamp(float(lo), float(hi))

    def _get_iter_mix(self, stage_idx: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self.pool_iter_anchor_mix or self.pool_iter_mix is None:
            return torch.tensor(0.0, device=device, dtype=dtype)
        if self.pool_iter_anchor_mix_mode == "scalar":
            return torch.sigmoid(self.pool_iter_mix.to(device=device, dtype=dtype))
        return torch.sigmoid(self.pool_iter_mix.to(device=device, dtype=dtype)[stage_idx])

    def _pool_readout(self, x: torch.Tensor, anchor: torch.Tensor, tau: torch.Tensor | None = None) -> torch.Tensor:
        feat = x.flatten(2)
        anchor_n = F.normalize(anchor, dim=-1)
        feat_n = F.normalize(feat, dim=1)

        if tau is None:
            tau = self.attn_temp
        tau = tau.clamp(0.1, 10.0)

        logits = torch.bmm(anchor_n, feat_n) / tau
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
        attn = torch.softmax(logits, dim=-1)
        pooled = torch.bmm(attn, feat.transpose(1, 2).contiguous())
        return self.out_norm(pooled + anchor)

    def _pool_readout_enhanced(self, stage_maps: List[torch.Tensor], anchor: torch.Tensor) -> torch.Tensor:
        if self.pool_mode == "single":
            return self._pool_readout(stage_maps[-1], anchor)

        if self.pool_mode == "iterative":
            q = anchor
            anchor0 = anchor
            for si, xs in enumerate(stage_maps):
                tau = self._get_iter_tau(si, device=anchor.device, dtype=anchor.dtype)
                q_new = self._pool_readout(xs, q, tau=tau)
                if self.pool_iter_anchor_mix:
                    g = self._get_iter_mix(si, device=anchor.device, dtype=anchor.dtype)
                    q = self.out_norm((1.0 - g) * q_new + g * anchor0)
                else:
                    q = q_new
            return q

        if self.pool_mode == "ms_late":
            q_list = [self._pool_readout(xs, anchor) for xs in stage_maps]
            w = self._pool_stage_weights(device=anchor.device, dtype=anchor.dtype)
            q = 0.0
            for si, qi in enumerate(q_list):
                q = q + w[si].view(1, 1, 1) * qi
            return self.out_norm(q)

        raise ValueError(f"Unsupported pool_mode={self.pool_mode}")

    def forward(self, feats_by_level: Dict[int, torch.Tensor], query_anchor: torch.Tensor):
        stage_maps = []
        lv0 = self.fuse_levels[0]
        x = self.proj[str(lv0)](feats_by_level[lv0])
        x = self.refine[0](x)
        stage_maps.append(x)

        for i, lv in enumerate(self.fuse_levels[1:], start=1):
            cur = self.proj[str(lv)](feats_by_level[lv])
            x = F.interpolate(x, size=cur.shape[-2:], mode="bilinear", align_corners=self.align_corners)
            x = x + cur
            x = self.refine[i](x)
            stage_maps.append(x)

        b = x.shape[0]
        anchor = query_anchor.expand(b, -1, -1).contiguous()
        return self._pool_readout_enhanced(stage_maps, anchor)
