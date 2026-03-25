import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _squeeze_hw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4 and x.shape[1] == 1:
        return x[:, 0]
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected [B,1,H,W] or [B,H,W], got {tuple(x.shape)}")


@torch.no_grad()
def _threshold_accuracy_count(pred: torch.Tensor, label: torch.Tensor, gamma: int = 1) -> float:
    eps = 1e-8
    ratio = torch.maximum(label / (pred + eps), pred / (label + eps))
    thresh = 1.25 ** gamma
    return float((ratio < thresh).float().sum().item())


@torch.no_grad()
def _mre_sum(pred: torch.Tensor, label: torch.Tensor) -> float:
    eps = 1e-8
    return float(torch.abs((pred - label) / (label + eps)).sum().item())


class RegressionMeter:
    def __init__(self, fp_thresh: float = 2.0):
        self.fp_thresh = float(fp_thresh)
        self.reset()

    def reset(self):
        self.acc = np.zeros(11, dtype=np.float64)

    @torch.no_grad()
    def update(self, pred_height: torch.Tensor, gt_height: torch.Tensor, valid_mask: torch.Tensor):
        pred = _squeeze_hw(pred_height).float()
        gt = _squeeze_hw(gt_height).float()
        vm = _squeeze_hw(valid_mask).to(torch.bool)

        pixels_valid = int(vm.sum().item())
        if pixels_valid > 0:
            pred_all = pred[vm].clamp_min(0.0)
            gt_all = gt[vm]
            diff_all = gt_all - pred_all
            sq_error_all = float((diff_all * diff_all).sum().item())
        else:
            sq_error_all = 0.0

        region = vm & (gt > self.fp_thresh)
        pixels_fp = int(region.sum().item())
        if pixels_fp <= 0:
            self.acc[6] += pixels_valid
            self.acc[10] += sq_error_all
            return

        pred_sel = pred[region].clamp_min(0.0)
        gt_sel = gt[region]
        diff = gt_sel - pred_sel
        self.acc += np.array([
            float((diff * diff).sum().item()),
            float(diff.abs().sum().item()),
            _mre_sum(pred_sel, gt_sel),
            _threshold_accuracy_count(pred_sel, gt_sel, 1),
            _threshold_accuracy_count(pred_sel, gt_sel, 2),
            _threshold_accuracy_count(pred_sel, gt_sel, 3),
            pixels_valid,
            pixels_fp,
            float(gt_sel.sum().item()),
            float((gt_sel * gt_sel).sum().item()),
            sq_error_all,
        ], dtype=np.float64)

    def compute(self) -> Dict[str, Any]:
        a = self.acc
        pixels_valid = int(a[6])
        pixels_fp = int(a[7])
        out = {"Num_Pixels": pixels_valid, "Num_Pixels_FP": int(pixels_fp)}
        out["RMSE_all"] = float(np.sqrt(a[10] / pixels_valid)) if pixels_valid > 0 else 0.0
        if pixels_fp <= 0:
            out.update({"MRE": 0.0, "MAE": 0.0, "RMSE": 0.0, "Sigma_1": 0.0, "Sigma_2": 0.0, "Sigma_3": 0.0, "R2": 0.0})
            return out
        rmse = float(np.sqrt(a[0] / pixels_fp))
        mae = float(a[1] / pixels_fp)
        mre = float(a[2] / pixels_fp)
        s1 = float(a[3] / pixels_fp)
        s2 = float(a[4] / pixels_fp)
        s3 = float(a[5] / pixels_fp)
        denom = (a[9] - a[8] * a[8] / pixels_fp)
        r2 = float(1.0 - a[0] / denom) if denom > 1e-12 else 0.0
        out.update({"MRE": mre, "MAE": mae, "RMSE": rmse, "Sigma_1": s1, "Sigma_2": s2, "Sigma_3": s3, "R2": r2})
        return out


class SegmentationMeter:
    def __init__(self, threshold: float = 0.5, from_logits: bool = True):
        self.threshold = float(threshold)
        self.from_logits_default = bool(from_logits)
        self.reset()

    def reset(self):
        self.acc = np.zeros(4, dtype=np.float64)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, label: torch.Tensor, valid_mask: torch.Tensor, from_logits: Optional[bool] = None):
        if from_logits is None:
            from_logits = self.from_logits_default
        vm = _squeeze_hw(valid_mask).to(torch.bool)
        lab = _squeeze_hw(label)
        if lab.dtype != torch.bool:
            lab = (lab > 0.5)
        pr = _squeeze_hw(pred).float()
        prob = torch.sigmoid(pr) if from_logits else pr.clamp(0, 1)
        pred_bin = (prob > self.threshold)
        tp = int((pred_bin & lab & vm).sum().item())
        fp = int((pred_bin & (~lab) & vm).sum().item())
        fn = int(((~pred_bin) & lab & vm).sum().item())
        pixels = int(vm.sum().item())
        self.acc += np.array([tp, fp, fn, pixels], dtype=np.float64)

    def compute(self) -> Dict[str, Any]:
        tp, fp, fn, pixels = self.acc
        precision = float((tp + 1e-8) / (tp + fp + 1e-8)) if tp > 0 else 0.0
        iou = float((tp + 1e-8) / (tp + fp + fn + 1e-8)) if tp > 0 else 0.0
        recall = float((tp + 1e-8) / (tp + fn + 1e-8)) if tp > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall + 1e-8))
        return {"Precision": precision, "IoU": iou, "Recall": recall, "F1-Score": f1, "Num_Pixels": int(pixels)}


class BinClassMeter:
    def __init__(self, fp_thresh: float = 2.0, building_only: bool = True):
        self.fp_thresh = float(fp_thresh)
        self.building_only = bool(building_only)
        self.reset()

    def reset(self):
        self.n = 0
        self.correct_pm1 = 0
        self.abs_err_sum_hard = 0.0
        self.abs_err_sum_exp = 0.0

    @staticmethod
    @torch.no_grad()
    def _bin_idx_from_edges(gt_height: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        gt = _squeeze_hw(gt_height).float()
        b, _, _ = gt.shape
        if edges.ndim == 1:
            edges = edges.view(1, -1).expand(b, -1)
        k = edges.shape[1] - 1
        idx_list = []
        for bi in range(b):
            eb = edges[bi].to(device=gt.device, dtype=gt.dtype)
            idx = torch.bucketize(gt[bi], eb, right=False) - 1
            idx_list.append(idx.clamp(min=0, max=k - 1))
        return torch.stack(idx_list, dim=0).long()

    @torch.no_grad()
    def update(self, height_bin_logits: torch.Tensor, gt_height: torch.Tensor, valid_mask: torch.Tensor, bin_edges: torch.Tensor):
        device = height_bin_logits.device
        gt_height = gt_height.to(device)
        m = _squeeze_hw(valid_mask).to(device=device, dtype=torch.bool)
        if self.building_only:
            m = m & (_squeeze_hw(gt_height) > self.fp_thresh)
        cnt = int(m.sum().item())
        if cnt <= 0:
            return

        bin_gt = self._bin_idx_from_edges(gt_height, bin_edges.to(device))
        pred_hard = torch.argmax(height_bin_logits, dim=1)
        diff_hard = (pred_hard[m] - bin_gt[m]).abs()
        self.correct_pm1 += int((diff_hard <= 1).sum().item())
        self.abs_err_sum_hard += float(diff_hard.float().sum().item())

        p = torch.softmax(height_bin_logits, dim=1)
        k = p.shape[1]
        idx = torch.arange(k, device=device, dtype=p.dtype).view(1, k, 1, 1)
        ebin = (p * idx).sum(dim=1)
        diff_exp = (ebin[m] - bin_gt[m].to(ebin.dtype)).abs()
        self.abs_err_sum_exp += float(diff_exp.sum().item())
        self.n += cnt

    def compute(self) -> Dict[str, Any]:
        if self.n <= 0:
            return {"bin_acc_pm1": 0.0, "bin_mae": 0.0, "bin_mae_hard": 0.0, "ebin_mae": 0.0, "bin_n": 0}
        bin_mae_hard = float(self.abs_err_sum_hard / self.n)
        ebin_mae = float(self.abs_err_sum_exp / self.n)
        return {"bin_acc_pm1": float(self.correct_pm1 / self.n), "bin_mae": bin_mae_hard, "bin_mae_hard": bin_mae_hard, "ebin_mae": ebin_mae, "bin_n": int(self.n)}


def merge_metrics(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for d in dicts:
        if d:
            out.update(d)
    return out


@dataclass
class MetricsBundle:
    mode: str
    fp_thresh: float = 2.0
    seg_threshold: float = 0.5
    enable_bin_class: bool = False
    seg_from_logits: bool = True
    fp_temp: float = 1.0
    seg_source: str = "height"
    seg_return: str = "binary"

    def __post_init__(self):
        self.mode = str(self.mode)
        self.seg_meter = SegmentationMeter(threshold=self.seg_threshold, from_logits=self.seg_from_logits)
        self.reg_meter = RegressionMeter(fp_thresh=self.fp_thresh) if self.mode in ("reg", "multi") else None
        self.bin_meter = BinClassMeter(fp_thresh=self.fp_thresh) if self.enable_bin_class else None

    def reset(self):
        self.seg_meter.reset()
        if self.reg_meter is not None:
            self.reg_meter.reset()
        if self.bin_meter is not None:
            self.bin_meter.reset()

    @staticmethod
    def pred_height_from_output(output: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "height" in output:
            return output["height"]
        if ("height_bin_logits" not in output) or ("bin_edges" not in output):
            raise KeyError('Cannot derive height prediction from model output.')
        logits = output["height_bin_logits"]
        edges = output["bin_edges"]
        if edges.ndim == 1:
            edges = edges.view(1, -1).expand(logits.shape[0], -1)
        centers = 0.5 * (edges[:, :-1] + edges[:, 1:])
        p = torch.softmax(logits, dim=1)
        return (p * centers[:, :, None, None]).sum(dim=1, keepdim=True)

    @staticmethod
    def pred_seg_from_output(output: Dict[str, torch.Tensor], fp_thresh: float = 2.0, seg_threshold: float = 0.5, source: str = "height", return_type: str = "binary") -> torch.Tensor:
        src = str(source).lower()
        if src == "fp_logits" and "fp_logits" in output:
            logits = output["fp_logits"]
            prob = torch.sigmoid(logits)
        else:
            h = MetricsBundle.pred_height_from_output(output)
            logits = h - fp_thresh
            prob = torch.sigmoid(logits)
        if return_type == "logits":
            return logits
        if return_type == "prob":
            return prob
        if return_type == "binary":
            return (prob >= seg_threshold).float()
        raise ValueError(f"Unknown return_type={return_type}")

    @torch.no_grad()
    def update(self, output: Dict[str, torch.Tensor], gt_height: torch.Tensor, valid_mask: torch.Tensor, seg_gt: Optional[torch.Tensor] = None):
        pred_height = None
        if self.mode in ("reg", "multi"):
            pred_height = self.pred_height_from_output(output)
        if self.mode == "multi":
            seg_pred = self.pred_seg_from_output(output, self.fp_thresh, self.seg_threshold, source=self.seg_source, return_type=self.seg_return)
            seg_gt_ = (gt_height > self.fp_thresh).float() if seg_gt is None else seg_gt
            self.seg_meter.update(seg_pred, seg_gt_, valid_mask, from_logits=(self.seg_return == "logits"))
            self.reg_meter.update(pred_height, gt_height, valid_mask)
        elif self.mode == "reg":
            seg_pred = (pred_height > self.fp_thresh).float()
            seg_gt_ = (gt_height > self.fp_thresh).float() if seg_gt is None else seg_gt
            self.seg_meter.update(seg_pred, seg_gt_, valid_mask, from_logits=False)
            self.reg_meter.update(pred_height, gt_height, valid_mask)
        else:
            seg_pred = self.pred_seg_from_output(output, self.fp_thresh, self.seg_threshold, source=self.seg_source, return_type=self.seg_return)
            seg_gt_ = gt_height if seg_gt is None else seg_gt
            self.seg_meter.update(seg_pred, seg_gt_, valid_mask, from_logits=(self.seg_return == "logits"))

        if self.bin_meter is not None and "height_bin_logits" in output and "bin_edges" in output:
            self.bin_meter.update(output["height_bin_logits"], gt_height, valid_mask, output["bin_edges"])

    def compute(self) -> Dict[str, Any]:
        seg_dict = self.seg_meter.compute()
        reg_dict = self.reg_meter.compute() if self.reg_meter is not None else {}
        bin_dict = self.bin_meter.compute() if self.bin_meter is not None else {}
        return merge_metrics(seg_dict, reg_dict, bin_dict)
