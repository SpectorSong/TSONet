from typing import Sequence, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import DoubleConv


class LRGT(nn.Module):
    """
    Lightweight Residual Group-wise Trunk.
    Stronger trunk: 2~4 stacked residual blocks.
    Each block: group 1x1 -> GELU -> DWConv3x3 -> GELU -> group 1x1, with residual + learnable scale(init 0).
    NOTE: Use an inner nn.Module for blocks to safely hold Parameters (no ModuleDict scale bug).
    """

    class _ResBlock(nn.Module):
        def __init__(self, ch: int, groups_1x1: int = 4, use_bn: bool = True, dw_kernel: int = 3):
            super().__init__()
            ch = int(ch)
            g = int(groups_1x1)
            if g < 1:
                g = 1
            if ch % g != 0:
                g = 1

            pad = dw_kernel // 2
            self.use_bn = bool(use_bn)

            self.pw1 = nn.Conv2d(ch, ch, kernel_size=1, groups=g, bias=not self.use_bn)
            self.bn1 = nn.BatchNorm2d(ch) if self.use_bn else nn.Identity()

            self.dw = nn.Conv2d(ch, ch, kernel_size=dw_kernel, padding=pad, groups=ch, bias=not self.use_bn)
            self.bn2 = nn.BatchNorm2d(ch) if self.use_bn else nn.Identity()

            self.pw2 = nn.Conv2d(ch, ch, kernel_size=1, groups=g, bias=True)

            # init: make residual branch start near 0
            nn.init.kaiming_normal_(self.pw1.weight, nonlinearity="linear")
            nn.init.kaiming_normal_(self.dw.weight, nonlinearity="linear")
            nn.init.zeros_(self.pw2.weight)
            if self.pw2.bias is not None:
                nn.init.zeros_(self.pw2.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            r = F.gelu(self.bn1(self.pw1(x)))
            r = F.gelu(self.bn2(self.dw(r)))
            r = self.pw2(r)
            return x + r

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        hidden_ch: int = 64,
        depth: int = 3,          # clamp to [2,4]
        groups_1x1: int = 4,
        dw_kernel: int = 3,
        use_bn: bool = True,
    ):
        super().__init__()
        in_ch = int(in_ch)
        out_ch = int(out_ch)
        hidden_ch = int(hidden_ch)

        depth = int(depth)
        depth = max(2, min(depth, 4))

        g = int(groups_1x1)
        if g < 1:
            g = 1
        if hidden_ch % g != 0:
            g = 1

        self.use_bn = bool(use_bn)

        self.proj_in = nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=not self.use_bn)
        self.bn_in = nn.BatchNorm2d(hidden_ch) if self.use_bn else nn.Identity()

        self.blocks = nn.Sequential(*[
            LRGT._ResBlock(
                ch=hidden_ch,
                groups_1x1=g,
                use_bn=self.use_bn,
                dw_kernel=dw_kernel,
            )
            for _ in range(depth)
        ])

        self.proj_out = nn.Conv2d(hidden_ch, out_ch, kernel_size=1, bias=True)

        # init
        nn.init.kaiming_normal_(self.proj_in.weight, nonlinearity="linear")
        nn.init.zeros_(self.proj_out.weight)
        if self.proj_out.bias is not None:
            nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.bn_in(self.proj_in(x)))
        y = self.blocks(y)
        y = self.proj_out(y)
        return y


class CSEM(nn.Module):
    """
    Cross Steam Exchange Module.
      - dp: separable head (DW3x3 + PW1x1) -> update prob
      - ds: simplified head (1x1)          -> update fp   (simpler & cheaper)
      - spatial_conf: reliability OR conflict (switch), applied ONLY to dp (prob update)
      - boundary_mask optional: multiplies g (shared)
    """
    def __init__(
        self,
        c_prob: int,
        c_fp: int,
        hidden_ch: Optional[int] = None,
        trunk_depth: int = 3,          # 2~4
        groups_1x1: int = 4,
        use_bn: bool = True,
        gate_bias_init: float = -4.0,
        alpha_prob_init: float = 0.0,
        alpha_fp_init: float = 0.0,
        conf_bias_init: float = 3.0,
        conf_detach: bool = True,
    ):
        super().__init__()
        c_prob = int(c_prob)
        c_fp = int(c_fp)
        self.use_bn = bool(use_bn)
        self.conf_detach = bool(conf_detach)

        if hidden_ch is None:
            hidden_ch = max(32, (c_prob + c_fp) // 2)
        hidden_ch = int(hidden_ch)

        # shared trunk
        self.trunk = LRGT(
            in_ch=c_prob + c_fp,
            out_ch=hidden_ch,
            hidden_ch=hidden_ch,
            depth=int(trunk_depth),
            groups_1x1=int(groups_1x1),
            dw_kernel=3,
            use_bn=self.use_bn,
        )

        # spatial gate from z
        self.gate_head = nn.Conv2d(hidden_ch, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.gate_head.weight)
        if self.gate_head.bias is not None:
            nn.init.constant_(self.gate_head.bias, float(gate_bias_init))

        # ---- dp head: separable conv (DW + PW) -> c_prob ----
        self.dp_dw = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1, groups=hidden_ch, bias=not self.use_bn)
        self.dp_bn = nn.BatchNorm2d(hidden_ch) if self.use_bn else nn.Identity()
        self.dp_pw = nn.Conv2d(hidden_ch, c_prob, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.dp_dw.weight, nonlinearity="linear")
        nn.init.zeros_(self.dp_pw.weight)
        if self.dp_pw.bias is not None:
            nn.init.zeros_(self.dp_pw.bias)

        # ---- ds head: simplified (1x1) -> c_fp ----
        # keep near-identity: weight init 0
        self.ds_head = nn.Conv2d(hidden_ch, c_fp, kernel_size=1, bias=True)
        nn.init.zeros_(self.ds_head.weight)
        if self.ds_head.bias is not None:
            nn.init.zeros_(self.ds_head.bias)

        # ---- reliability spatial_conf head (DW(fp) + PW -> sigmoid) ----
        self.conf_dw = nn.Conv2d(c_fp, c_fp, kernel_size=3, padding=1, groups=c_fp, bias=not self.use_bn)
        self.conf_bn = nn.BatchNorm2d(c_fp) if self.use_bn else nn.Identity()
        self.conf_pw = nn.Conv2d(c_fp, 1, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.conf_dw.weight, nonlinearity="linear")
        nn.init.zeros_(self.conf_pw.weight)
        if self.conf_pw.bias is not None:
            nn.init.constant_(self.conf_pw.bias, float(conf_bias_init))

        # learnable strengths
        self.alpha_prob = nn.Parameter(torch.tensor(float(alpha_prob_init)))
        self.alpha_fp = nn.Parameter(torch.tensor(float(alpha_fp_init)))

    def _conf_spatial(self, prob_feat: torch.Tensor, fp_feat: torch.Tensor) -> torch.Tensor:
        """
        Return conf_sp in [0,1], shape [B,1,H,W].
        - reliability: from fp_feat only (DW+PW -> sigmoid)
        - conflict: based on structural disagreement between prob and fp (edge magnitude difference)
        """
        fp_in = fp_feat.detach() if self.conf_detach else fp_feat
        conf = torch.sigmoid(self.conf_pw(F.gelu(self.conf_bn(self.conf_dw(fp_in)))))
        return conf

    def forward(
        self,
        prob_feat: torch.Tensor,
        fp_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # shared interaction feature
        z = self.trunk(torch.cat([prob_feat, fp_feat], dim=1))

        # spatial gate
        g = torch.sigmoid(self.gate_head(z))  # [B,1,H,W]

        # spatial conf (reliability/conflict), applied ONLY to prob update
        conf_sp = self._conf_spatial(prob_feat, fp_feat)  # [B,1,H,W]

        # contents
        dp = self.dp_pw(F.gelu(self.dp_bn(self.dp_dw(z))))  # [B,c_prob,H,W]
        ds = self.ds_head(z)                                # [B,c_fp,H,W]  (simplified)

        prob_feat = prob_feat + (self.alpha_prob * g * conf_sp * dp)
        fp_feat = fp_feat + (self.alpha_fp * g * ds)  # no conf here

        return prob_feat, fp_feat


class HierarchicalUNetDualDecoder5(nn.Module):
    """

      f1: [B, d,   H,   W]
      f2: [B, 2d, H/2, W/2]
      f3: [B, 4d, H/4, W/4]
      f4: [B, 8d, H/8, W/8]
      f5: [B,16d, H/16,W/16]



      pyr: {1: pix, 2: d2, 3: d3, 4: d4, 5: d5}
    """

    def __init__(
        self,
        dim: int,
        out_dim: int = None,
        use_task_exchange: bool = False,
        exchange_levels: Sequence[int] = (5, 4, 3, 2, 1),
        exchange_on_fp_boundary: bool = False,
        fp_zone_kernel_size: int = 3,
        fp_zone_thresh: float = 0.5,
        exchange_alpha_prob_init: float = 0.0,
        exchange_alpha_fp_init: float = 0.0,
        exchange_conf_bias_init: float = 3.0,
    ):
        super().__init__()
        d = int(dim)
        self.dim = d
        self.out_dim = int(out_dim) if out_dim is not None else d

        # boundary options
        self.exchange_on_fp_boundary = bool(exchange_on_fp_boundary)
        self.fp_zone_kernel_size = int(fp_zone_kernel_size)
        self.fp_zone_thresh = float(fp_zone_thresh)

        self.prob_proj5 = nn.Conv2d(16 * d, 8 * d, kernel_size=1, bias=False)
        self.fp_proj5   = nn.Conv2d(16 * d, 8 * d, kernel_size=1, bias=False)

        # --- up blocks ---
        self.prob_up4 = DoubleConv(8*d + 8*d, 8*d)
        self.prob_up3 = DoubleConv(8*d + 4*d, 4*d)
        self.prob_up2 = DoubleConv(4*d + 2*d, 2*d)
        self.prob_up1 = DoubleConv(2*d + 1*d, 1*d)

        self.fp_up4 = DoubleConv(8*d + 8*d, 8*d)
        self.fp_up3 = DoubleConv(8*d + 4*d, 4*d)
        self.fp_up2 = DoubleConv(4*d + 2*d, 2*d)
        self.fp_up1 = DoubleConv(2*d + 1*d, 1*d)

        self.prob_out_proj = nn.Conv2d(d, self.out_dim, kernel_size=1, bias=True)
        self.fp_out_proj   = nn.Conv2d(d, self.out_dim, kernel_size=1, bias=True)

        # exchange blocks
        self.use_task_exchange = bool(use_task_exchange)
        self.exchange_levels = tuple(int(x) for x in exchange_levels)

        self.exchange = nn.ModuleDict()
        if self.use_task_exchange:
            c_by_level = {1: d, 2: 2*d, 3: 4*d, 4: 8*d, 5: 8*d}
            for lv in self.exchange_levels:
                c = int(c_by_level[int(lv)])
                self.exchange[str(int(lv))] = CSEM(
                    c_prob=c,
                    c_fp=c,
                    trunk_depth=2,
                    groups_1x1=4,
                    use_bn=True,
                    gate_bias_init=-4.0,
                    alpha_prob_init=exchange_alpha_prob_init,
                    alpha_fp_init=exchange_alpha_fp_init,
                    conf_bias_init=exchange_conf_bias_init,
                    conf_detach=True
                )

        self.fp_side_heads = nn.ModuleDict({
            "5": nn.Conv2d(8 * d, 1, kernel_size=1, bias=True),
            "4": nn.Conv2d(8 * d, 1, kernel_size=1, bias=True),
            "3": nn.Conv2d(4 * d, 1, kernel_size=1, bias=True),
            "2": nn.Conv2d(2 * d, 1, kernel_size=1, bias=True),
            "1": nn.Conv2d(1 * d, 1, kernel_size=1, bias=True),
        })

    @staticmethod
    def _upsample_to(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def _maybe_exchange(self, prob_feat: torch.Tensor, fp_feat: torch.Tensor, lv: int):
        key = str(int(lv))
        if (not self.use_task_exchange) or (key not in self.exchange):
            return prob_feat, fp_feat
        return self.exchange[key](prob_feat, fp_feat)

    def forward(self, f1, f2, f3, f4, f5):
        # d5
        p5 = self.prob_proj5(f5)
        s5 = self.fp_proj5(f5)
        p5, s5 = self._maybe_exchange(p5, s5, lv=5)

        # d4
        x = self._upsample_to(p5, f4)
        p4 = self.prob_up4(torch.cat([x, f4], dim=1))
        x = self._upsample_to(s5, f4)
        s4 = self.fp_up4(torch.cat([x, f4], dim=1))
        p4, s4 = self._maybe_exchange(p4, s4, lv=4)

        # d3
        x = self._upsample_to(p4, f3)
        p3 = self.prob_up3(torch.cat([x, f3], dim=1))
        x = self._upsample_to(s4, f3)
        s3 = self.fp_up3(torch.cat([x, f3], dim=1))
        p3, s3 = self._maybe_exchange(p3, s3, lv=3)

        # d2
        x = self._upsample_to(p3, f2)
        p2 = self.prob_up2(torch.cat([x, f2], dim=1))
        x = self._upsample_to(s3, f2)
        s2 = self.fp_up2(torch.cat([x, f2], dim=1))
        p2, s2 = self._maybe_exchange(p2, s2, lv=2)

        # d1
        x = self._upsample_to(p2, f1)
        p1 = self.prob_up1(torch.cat([x, f1], dim=1))
        x = self._upsample_to(s2, f1)
        s1 = self.fp_up1(torch.cat([x, f1], dim=1))
        p1, s1 = self._maybe_exchange(p1, s1, lv=1)

        prob_pyr = {1: p1, 2: p2, 3: p3, 4: p4, 5: p5}
        fp_pyr   = {1: s1,   2: s2, 3: s3, 4: s4, 5: s5}
        return p1, prob_pyr, s1, fp_pyr


class HierarchicalUNetProbDecoder5(nn.Module):
    """
    Prob-only version of HierarchicalUNetDualDecoder5 (no fp path, no task exchange).

    Output semantics:
      - prob_pix: [B, dim, H, W] (same as dual decoder's prob_pix)
      - prob_pyr: {1..5} multi-scale features
    """
    def __init__(self, dim: int, out_dim: int = None):
        super().__init__()
        d = int(dim)
        self.dim = d
        self.out_dim = int(out_dim) if out_dim is not None else d

        # bottleneck: 16d -> 8d
        self.prob_proj5 = nn.Conv2d(16 * d, 8 * d, kernel_size=1, bias=False)

        # up blocks
        self.prob_up4 = DoubleConv(8 * d + 8 * d, 8 * d)
        self.prob_up3 = DoubleConv(8 * d + 4 * d, 4 * d)
        self.prob_up2 = DoubleConv(4 * d + 2 * d, 2 * d)
        self.prob_up1 = DoubleConv(2 * d + 1 * d, 1 * d)

        # keep the same style as UNetPixelDecoder5
        self.prob_out_proj = nn.Conv2d(d, self.out_dim, kernel_size=1, bias=True)

    @staticmethod
    def _upsample_to(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, f1, f2, f3, f4, f5):
        # d5
        p5 = self.prob_proj5(f5)

        # d4
        x = self._upsample_to(p5, f4)
        p4 = self.prob_up4(torch.cat([x, f4], dim=1))

        # d3
        x = self._upsample_to(p4, f3)
        p3 = self.prob_up3(torch.cat([x, f3], dim=1))

        # d2
        x = self._upsample_to(p3, f2)
        p2 = self.prob_up2(torch.cat([x, f2], dim=1))

        # d1
        x = self._upsample_to(p2, f1)
        p1 = self.prob_up1(torch.cat([x, f1], dim=1))

        prob_pyr = {1: p1, 2: p2, 3: p3, 4: p4, 5: p5}
        return p1, prob_pyr
