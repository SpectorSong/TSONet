from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import UNetEncoder5
from .decoder import HierarchicalUNetDualDecoder5, HierarchicalUNetProbDecoder5
from .bins_head import FEBR, MLP, SinePositionalEncoding2D, BinsFormerDetrDecoder


class TSONet(nn.Module):
    """Two-Stream Ordinal Network (TSONet).

    The released model keeps the final FEBR head by default and also retains the
    BinsFormer-style DETR head for the FEBR ablation. The direct height head is
    only used when the bin head is disabled by ablation.
    """

    def __init__(
        self,
        opt,
        dim: int = 32,
        hidden_dim: int = 256,
        n_bins: int = 64,
        mem_levels: Sequence[int] = (5, 4, 3),
        ffn_expansion_factor: float = 2.0,
        bias: bool = False,
        LayerNorm_type: str = "WithBias",
    ):
        super().__init__()
        self.opt = opt

        self.n_bins = int(getattr(opt, "num_height_bins", n_bins))
        self.h_min = float(getattr(opt, "h_min", 0.0))
        self.h_max = float(getattr(opt, "h_max", 145.0))
        if self.h_max <= self.h_min:
            raise ValueError(f"h_max must be > h_min, got h_min={self.h_min}, h_max={self.h_max}")

        in_ch = int(getattr(opt, "in_channels", 7))
        self.dim = int(getattr(opt, "dim", dim))
        self.hidden_dim = int(getattr(opt, "hidden_dim", hidden_dim))

        self.encoder = UNetEncoder5(in_channels=in_ch, dim=self.dim, bias=bias, norm="bn")

        _mlv = getattr(opt, "mem_levels", mem_levels)
        if isinstance(_mlv, str):
            _mlv = tuple(int(x) for x in _mlv.split(",") if x.strip())
        elif isinstance(_mlv, int):
            _mlv = (_mlv,)
        self.mem_levels = tuple(int(x) for x in _mlv)

        self.pix_c_by_level: Dict[int, int] = {
            1: self.dim,
            2: 2 * self.dim,
            3: 4 * self.dim,
            4: 8 * self.dim,
            5: 8 * self.dim,
        }

        # task exchange config
        self.use_task_exchange = bool(getattr(opt, "use_task_exchange", False))
        self.exchange_on_fp_boundary = bool(getattr(opt, "exchange_on_fp_boundary", False))
        self.fp_zone_kernel_size = int(getattr(opt, "fp_zone_kernel_size", 3))
        self.fp_zone_thresh = float(getattr(opt, "fp_zone_thresh", 0.5))
        alpha0 = float(getattr(opt, "exchange_alpha_init", 0.01))
        self.exchange_alpha_prob_init = float(getattr(opt, "exchange_alpha_prob_init", alpha0))
        self.exchange_alpha_fp_init = float(getattr(opt, "exchange_alpha_fp_init", alpha0))
        self.exchange_conf_bias_init = float(getattr(opt, "exchange_conf_bias_init", 3.0))

        _ex_lv = getattr(opt, "exchange_levels", "5,4,3,2,1")
        if isinstance(_ex_lv, str):
            exchange_levels = tuple(int(x) for x in _ex_lv.split(",") if x.strip())
        elif isinstance(_ex_lv, int):
            exchange_levels = (_ex_lv,)
        else:
            exchange_levels = tuple(int(x) for x in _ex_lv)

        # ablation switches
        self.ablate_no_fp = bool(getattr(opt, "ablate_no_fp", False))
        self.ablate_no_bins = bool(getattr(opt, "ablate_no_bins", False))
        self.enable_fp_branch = not self.ablate_no_fp
        self.enable_bin_head = not self.ablate_no_bins

        self.prob_height_head = self.ablate_no_bins
        self.prob_height_head_act = str(getattr(opt, "prob_height_head_act", "linear")).lower()
        if self.prob_height_head_act not in ("linear", "sigmoid_range"):
            raise ValueError(
                f"prob_height_head_act must be linear/sigmoid_range, got {self.prob_height_head_act}"
            )

        # decoders
        self.prob_decoder = None
        self.dual_decoder = None
        self.fp_out = None
        if self.enable_fp_branch:
            self.dual_decoder = HierarchicalUNetDualDecoder5(
                dim=self.dim,
                out_dim=self.dim,
                use_task_exchange=self.use_task_exchange,
                exchange_levels=exchange_levels,
                exchange_on_fp_boundary=self.exchange_on_fp_boundary,
                fp_zone_kernel_size=self.fp_zone_kernel_size,
                fp_zone_thresh=self.fp_zone_thresh,
                exchange_alpha_prob_init=self.exchange_alpha_prob_init,
                exchange_alpha_fp_init=self.exchange_alpha_fp_init,
                exchange_conf_bias_init=self.exchange_conf_bias_init,
            )
            self.fp_out = nn.Conv2d(self.dim, 1, kernel_size=3, padding=1, bias=True)
        else:
            self.use_task_exchange = False
            self.prob_decoder = HierarchicalUNetProbDecoder5(dim=self.dim, out_dim=self.dim)

        self.prob_height_out = None
        if self.prob_height_head:
            self.prob_height_out = nn.Conv2d(self.dim, 1, kernel_size=3, padding=1, bias=True)

        self.query_mode = str(getattr(opt, "query_mode", "febr")).lower()
        refine_blocks = getattr(opt, "febr_blocks", (1, 1, 1))
        ms_pool_mode = str(getattr(opt, "ms_pool_mode", "single"))
        ms_pool_fuse_learnable = bool(getattr(opt, "ms_pool_fuse_learnable", True))
        pool_iter_anchor_mix = bool(getattr(opt, "pool_iter_anchor_mix", False))
        pool_iter_anchor_mix_mode = str(getattr(opt, "pool_iter_anchor_mix_mode", "per_stage"))
        pool_iter_anchor_mix_init = float(getattr(opt, "pool_iter_anchor_mix_init", -2.0))
        pool_iter_tau_mode = str(getattr(opt, "pool_iter_tau_mode", "shared"))
        pool_iter_tau_init = float(getattr(opt, "pool_iter_tau_init", 1.0))
        pool_iter_tau_clamp = getattr(opt, "pool_iter_tau_clamp", (0.1, 10.0))
        if isinstance(pool_iter_tau_clamp, str):
            pool_iter_tau_clamp = tuple(float(x) for x in pool_iter_tau_clamp.split(",") if x.strip())
        pool_iter_tau_clamp = (float(pool_iter_tau_clamp[0]), float(pool_iter_tau_clamp[1]))

        self.ms_query_head = None
        self.detr_decoder = None
        self.detr_mem_proj = None
        self.detr_level_embed = None
        self.detr_query_pe = None
        self.detr_pos_enc = None
        self.width_mlp = None
        self.mask_feat_proj = None
        self.mask_mlp = None
        self.logit_scale = None
        self.query_feat = None

        if self.enable_bin_head:
            if self.query_mode not in ("febr", "detr"):
                raise ValueError(f"query_mode must be 'febr' or 'detr', got {self.query_mode}")

            if self.query_mode == "detr":
                self.detr_pos_enc = SinePositionalEncoding2D(num_feats=self.hidden_dim // 2, normalize=True)
                self.detr_mem_proj = nn.ModuleList()
                self.detr_level_embed = nn.ParameterList()
                for lv in self.mem_levels:
                    in_ch_lv = int(self.pix_c_by_level[int(lv)])
                    self.detr_mem_proj.append(nn.Conv2d(in_ch_lv, self.hidden_dim, kernel_size=1, bias=False))
                    self.detr_level_embed.append(nn.Parameter(torch.zeros(1, 1, self.hidden_dim)))

                self.detr_query_pe = nn.Parameter(torch.zeros(1, self.n_bins, self.hidden_dim))
                nn.init.normal_(self.detr_query_pe, std=0.02)
                for p in self.detr_level_embed:
                    nn.init.zeros_(p)

                decoder_heads = int(getattr(opt, "decoder_heads", 8))
                decoder_ffn_dim = int(getattr(opt, "decoder_ffn_dim", 1024))
                decoder_operation = str(getattr(opt, "decoder_operation", "//"))
                decoder_num_blocks = getattr(opt, "decoder_num_blocks", None)
                if decoder_num_blocks is None:
                    decoder_num_blocks = tuple(1 for _ in self.mem_levels)
                if isinstance(decoder_num_blocks, str):
                    decoder_num_blocks = tuple(int(x) for x in decoder_num_blocks.split(",") if x.strip())
                elif isinstance(decoder_num_blocks, int):
                    decoder_num_blocks = (int(decoder_num_blocks),)
                if len(decoder_num_blocks) != len(self.mem_levels):
                    raise ValueError(
                        f"decoder_num_blocks length must match mem_levels length: {len(decoder_num_blocks)} vs {len(self.mem_levels)}"
                    )
                self.detr_decoder = BinsFormerDetrDecoder(
                    dim=self.hidden_dim,
                    num_heads=decoder_heads,
                    ffn_dim=decoder_ffn_dim,
                    blocks_per_level=tuple(int(x) for x in decoder_num_blocks),
                    operation=decoder_operation,
                    dropout=0.0,
                )
            else:
                self.ms_query_head = FEBR(
                    in_channels_by_level=self.pix_c_by_level,
                    fuse_levels=self.mem_levels,
                    hidden_dim=self.hidden_dim,
                    n_bins=self.n_bins,
                    refine_blocks=refine_blocks,
                    heads=int(getattr(opt, "ms_heads", 8)),
                    ffn_expansion_factor=float(ffn_expansion_factor),
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    align_corners=True,
                    pool_mode=ms_pool_mode,
                    pool_fuse_learnable=ms_pool_fuse_learnable,
                    pool_iter_anchor_mix=pool_iter_anchor_mix,
                    pool_iter_anchor_mix_mode=pool_iter_anchor_mix_mode,
                    pool_iter_anchor_mix_init=pool_iter_anchor_mix_init,
                    pool_iter_tau_mode=pool_iter_tau_mode,
                    pool_iter_tau_init=pool_iter_tau_init,
                    pool_iter_tau_clamp=pool_iter_tau_clamp,
                )

            self.width_mlp = MLP(self.hidden_dim, self.hidden_dim, 1, num_layers=3)
            self.mask_feat_proj = nn.Conv2d(self.dim, self.hidden_dim, kernel_size=1, bias=False)
            self.mask_mlp = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, num_layers=3)
            self.logit_scale = nn.Parameter(torch.tensor(float(getattr(opt, "logit_scale_init", 10.0))))
            self.query_feat = nn.Parameter(torch.zeros(1, self.n_bins, self.hidden_dim))
            nn.init.normal_(self.query_feat, std=0.02)

    def _safe_softmax(self, x: torch.Tensor, dim: int, clamp_val: float = 50.0, eps: float = 1e-6) -> torch.Tensor:
        xf = torch.nan_to_num(x.float(), nan=0.0, posinf=clamp_val, neginf=-clamp_val).clamp(-clamp_val, clamp_val)
        p = torch.softmax(xf, dim=dim).clamp_min(eps)
        return p / p.sum(dim=dim, keepdim=True).clamp_min(eps)

    def _sanitize_edges(self, edges: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        ef = torch.nan_to_num(edges.float(), nan=self.h_min, posinf=self.h_max, neginf=self.h_min)
        ef[:, 0] = float(self.h_min)
        ef[:, -1] = float(self.h_max)
        ef = torch.maximum(ef, torch.cat([ef[:, :1], ef[:, :-1] + eps], dim=1))
        return ef

    def _widths_edges_from_q(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w_logits = self.width_mlp(q).squeeze(-1)
        w_ratio = self._safe_softmax(w_logits, dim=1, clamp_val=50.0, eps=1e-6)
        min_ratio = float(getattr(self.opt, "min_bin_ratio", 1e-4))
        if min_ratio > 0:
            w_ratio = w_ratio.clamp_min(min_ratio)
            w_ratio = w_ratio / w_ratio.sum(dim=1, keepdim=True).clamp_min(1e-6)

        w = w_ratio * float(self.h_max - self.h_min)
        w = torch.nan_to_num(w, nan=0.0, posinf=float(self.h_max - self.h_min), neginf=0.0).clamp_min(1e-6)
        edges = torch.cat(
            [
                torch.full((w.shape[0], 1), float(self.h_min), device=w.device, dtype=w.dtype),
                float(self.h_min) + torch.cumsum(w, dim=1),
            ],
            dim=1,
        )
        return w, self._sanitize_edges(edges)

    def _expected_height_from_bins(self, height_bin_logits: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
        if bin_edges.ndim == 1:
            bin_edges = bin_edges.view(1, -1).repeat(height_bin_logits.shape[0], 1)
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        p = self._safe_softmax(height_bin_logits, dim=1, clamp_val=50.0, eps=1e-6)
        return (p * centers[:, :, None, None]).sum(dim=1, keepdim=True)

    def _build_memory_levels_tokens_detr(self, prob_pyr: Dict[int, torch.Tensor]):
        mem_list, pos_list = [], []
        for proj, lvl_emb, lv in zip(self.detr_mem_proj, self.detr_level_embed, self.mem_levels):
            feat = prob_pyr[int(lv)]
            mem_2d = proj(feat)
            mem = mem_2d.flatten(2).transpose(1, 2).contiguous()
            pos = self.detr_pos_enc(mem_2d) + lvl_emb
            mem_list.append(mem)
            pos_list.append(pos)
        return mem_list, pos_list

    def _predict_bins_detr(self, mem_list, pos_list):
        batch = mem_list[0].shape[0]
        q = self.query_feat.expand(batch, -1, -1).contiguous()
        q_pos = self.detr_query_pe.expand(batch, -1, -1).contiguous()
        q = self.detr_decoder(q, q_pos, mem_list, pos_list)
        bin_widths, bin_edges = self._widths_edges_from_q(q)
        return q, bin_widths, bin_edges

    def forward(self, x, is_strong=None):
        _, f1, f2, f3, f4, f5 = self.encoder(x)

        if self.enable_fp_branch:
            prob_pix, prob_pyr, fp_pix, _ = self.dual_decoder(f1, f2, f3, f4, f5)
            fp_logits = self.fp_out(fp_pix)
        else:
            prob_pix, prob_pyr = self.prob_decoder(f1, f2, f3, f4, f5)
            fp_logits = None

        out = {}

        if self.enable_bin_head:
            if self.query_mode == "febr":
                q = self.ms_query_head(prob_pyr, self.query_feat)
                bin_widths, bin_edges = self._widths_edges_from_q(q)
            else:
                mem_list, pos_list = self._build_memory_levels_tokens_detr(prob_pyr)
                q, bin_widths, bin_edges = self._predict_bins_detr(mem_list, pos_list)

            mask_features = self.mask_feat_proj(prob_pix)
            feat_flat = mask_features.flatten(2)
            mask_embed = F.normalize(self.mask_mlp(q), dim=-1)
            logits_flat = torch.bmm(mask_embed, feat_flat)
            scale = self.logit_scale.clamp(0.0, 50.0)

            b, k, _ = logits_flat.shape
            h, w = mask_features.shape[-2:]
            height_bin_logits = (scale * logits_flat).view(b, k, h, w)
            height_bin_logits = torch.nan_to_num(height_bin_logits, nan=0.0, posinf=50.0, neginf=-50.0)
            height_pred = self._expected_height_from_bins(height_bin_logits, bin_edges)

            out.update({
                "height_bin_logits": height_bin_logits,
                "bin_widths": bin_widths,
                "bin_edges": bin_edges,
                "height": height_pred,
            })

        if self.prob_height_out is not None:
            h = self.prob_height_out(prob_pix)
            if self.prob_height_head_act == "sigmoid_range":
                h = torch.sigmoid(h) * float(self.h_max - self.h_min) + float(self.h_min)
            if self.enable_bin_head:
                out["height_prob"] = h
            else:
                out["height"] = h

        if fp_logits is not None:
            out["fp_logits"] = fp_logits

        return out
