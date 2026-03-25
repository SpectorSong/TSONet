from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_fp_zones(fp_mask: torch.Tensor, kernel_size: int = 3):
    """Split footprint into inner / boundary / background."""
    fp = fp_mask.float()
    fp_bin = (fp > 0.5).float()
    dilated = F.max_pool2d(fp_bin, kernel_size, stride=1, padding=kernel_size // 2)
    eroded = 1 - F.max_pool2d(1 - fp_bin, kernel_size, stride=1, padding=kernel_size // 2)
    boundary = (dilated - eroded).clamp(min=0.0)
    boundary = (boundary > 0.5).float()
    inner = (fp_bin - boundary).clamp(min=0.0)
    inner = (inner > 0.5).float()
    background = ((inner == 0) & (boundary == 0)).float()
    return inner, boundary, background


class RegressionLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.loss_name = str(opt.loss).lower()
        self.use_fp_weight = bool(getattr(opt, "use_fp_weight", False))
        self.fp_thresh = float(getattr(opt, "fp_thresh", 2.0))
        self.inner_w = float(getattr(opt, "inner_w", 1.0))
        self.boundary_w = float(getattr(opt, "boundary_w", 0.5))
        self.background_w = float(getattr(opt, "background_w", 0.3))

        if "smooth_l1" in self.loss_name:
            self.loss_fn = nn.SmoothL1Loss(reduction="none")
        elif "l2" in self.loss_name:
            self.loss_fn = nn.MSELoss(reduction="none")
        else:
            self.loss_fn = nn.L1Loss(reduction="none")

    def forward(self, pred: Dict[str, torch.Tensor], gt: torch.Tensor, mask: torch.Tensor, is_strong=None):
        pred_h = pred["height"]
        valid = (mask > 0.5).float()
        if valid.sum() <= 0:
            zero = pred_h.new_tensor(0.0)
            return zero, {"loss_h": zero.detach()}

        weight = valid
        if self.use_fp_weight:
            gt_fp = (gt > self.fp_thresh).float()
            inner, bd, bg = compute_fp_zones(gt_fp)
            weight = weight * (self.inner_w * inner + self.boundary_w * bd + self.background_w * bg)

        loss_map = self.loss_fn(pred_h, gt)
        loss_h = (loss_map * weight).sum() / weight.sum().clamp_min(1e-6)
        return loss_h, {"loss_h": loss_h.detach()}


class SegmentationLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.loss_spec = str(opt.loss).lower().replace(" ", "")
        self.use_fp_weight_seg = bool(getattr(opt, "use_fp_weight_seg", False))
        self.inner_w_seg = float(getattr(opt, "inner_w_seg", 1.0))
        self.boundary_w_seg = float(getattr(opt, "boundary_w_seg", 2.0))
        self.background_w_seg = float(getattr(opt, "background_w_seg", 0.5))
        self.components = []
        for part in self.loss_spec.split("+"):
            if not part:
                continue
            if "*" in part:
                name, w = part.split("*", 1)
                weight = float(w)
            else:
                name, weight = part, 1.0
            if name in {"bce", "wbce", "dice", "focal", "tver"}:
                self.components.append((name, weight))

    def _pixel_weight(self, gt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if not self.use_fp_weight_seg:
            return valid
        inner, bd, bg = compute_fp_zones(gt)
        reg_w = self.inner_w_seg * inner + self.boundary_w_seg * bd + self.background_w_seg * bg
        return valid * reg_w

    def _bce(self, logits, gt, weight):
        loss_map = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
        return (loss_map * weight).sum() / weight.sum().clamp_min(1e-6)

    def _wbce(self, logits, gt, weight):
        pos = (gt * (weight > 0)).sum()
        neg = ((1 - gt) * (weight > 0)).sum()
        pos_weight = (neg / (pos + 1e-6)).clamp(1.0, 100.0)
        loss_map = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
        posneg = torch.ones_like(gt)
        posneg[gt > 0.5] *= pos_weight
        w = weight * posneg
        return (loss_map * w).sum() / w.sum().clamp_min(1e-6)

    def _dice(self, logits, gt, weight):
        prob = torch.sigmoid(logits)
        inter = (weight * prob * gt).sum(dim=(1, 2, 3))
        union = (weight * prob).sum(dim=(1, 2, 3)) + (weight * gt).sum(dim=(1, 2, 3))
        dice = (2 * inter + 1e-6) / (union + 1e-6)
        return 1.0 - dice.mean()

    def _focal(self, logits, gt, weight, gamma: float = 2.0):
        prob = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
        pt = prob * gt + (1 - prob) * (1 - gt)
        loss_map = ((1 - pt) ** gamma) * ce
        return (loss_map * weight).sum() / weight.sum().clamp_min(1e-6)

    def _tversky(self, logits, gt, weight, alpha: float = 0.3, beta: float = 0.7):
        prob = torch.sigmoid(logits)
        tp = (weight * prob * gt).sum(dim=(1, 2, 3))
        fp = (weight * prob * (1 - gt)).sum(dim=(1, 2, 3))
        fn = (weight * (1 - prob) * gt).sum(dim=(1, 2, 3))
        score = (tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)
        return 1.0 - score.mean()

    def forward(self, pred: Dict[str, torch.Tensor], gt: torch.Tensor, mask: torch.Tensor, is_strong=None):
        logits = pred["fp_logits"]
        valid = (mask > 0.5).float()
        if valid.sum() <= 0:
            zero = logits.new_tensor(0.0)
            return zero, {"loss_seg": zero.detach()}

        weight = self._pixel_weight(gt, valid)
        loss = logits.new_tensor(0.0)
        for name, comp_w in self.components:
            if name == "bce":
                cur = self._bce(logits, gt, weight)
            elif name == "wbce":
                cur = self._wbce(logits, gt, weight)
            elif name == "dice":
                cur = self._dice(logits, gt, weight)
            elif name == "focal":
                cur = self._focal(logits, gt, weight)
            elif name == "tver":
                cur = self._tversky(logits, gt, weight)
            else:
                continue
            loss = loss + comp_w * cur
        return loss, {"loss_seg": loss.detach()}


class MultiTaskLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.reg_loss_fn = RegressionLoss(opt)
        self.seg_loss_fn = SegmentationLoss(opt)
        self.lambda_seg = float(getattr(opt, "lambda_seg", 1.0))
        self.fp_thresh = float(getattr(opt, "fp_thresh", 2.0))

    def update_epoch(self, epoch: int):
        return None

    def forward(self, pred: Dict[str, torch.Tensor], gt_h: torch.Tensor, mask: torch.Tensor, is_strong=None):
        loss_h, _ = self.reg_loss_fn(pred, gt_h, mask, is_strong)
        gt_fp = (gt_h > self.fp_thresh).float()
        if self.lambda_seg > 0 and "fp_logits" in pred:
            loss_seg, _ = self.seg_loss_fn(pred, gt_fp, mask, is_strong)
        else:
            loss_seg = loss_h.new_tensor(0.0)
        loss = loss_h + self.lambda_seg * loss_seg
        stats = {
            "loss_total": loss.detach(),
            "loss_h": loss_h.detach(),
            "loss_seg": (self.lambda_seg * loss_seg).detach() if self.lambda_seg > 0 and "fp_logits" in pred else None,
        }
        return loss, stats


def create_loss(opt):
    print(f"Using {opt.loss} loss!\n")
    if opt.mode == "multi":
        return MultiTaskLoss(opt)
    if opt.mode == "reg":
        return RegressionLoss(opt)
    if opt.mode == "seg":
        return SegmentationLoss(opt)
    raise NotImplementedError(f"Mode [{opt.mode}] is not implemented")
