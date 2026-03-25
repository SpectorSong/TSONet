import os
import math
import numpy as np
import rasterio
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image

import torch

from utils.losses import compute_fp_zones


@torch.no_grad()
def log_param_update_ratios(model, writer, step, param_snap):
    """
    groups: dict[group_name] = list of name keywords OR callable(name)->bool

    """
    groups = {
        "queries": ["query_feat", "query_pe"],
        "width_head": ["width_mlp."],
        "backbone": ["encoder.", "pixel_decoder."],  
    }

    def in_group(name, rule):
        if callable(rule):
            return rule(name)
        # list of keywords
        return any(k in name for k in rule)

    
    out = {g: {"d_sq": 0.0, "p_sq": 0.0, "cnt": 0} for g in groups}
    out["_all"] = {"d_sq": 0.0, "p_sq": 0.0, "cnt": 0}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        
        cur = p.detach()

        
        prev = param_snap.get(name, None)
        if prev is None:
            param_snap[name] = cur.clone()
            continue

        d = (cur - prev)
        
        param_snap[name] = cur.clone()

        d2 = d.float().pow(2).sum().item()
        p2 = cur.float().pow(2).sum().item()

        out["_all"]["d_sq"] += d2
        out["_all"]["p_sq"] += p2
        out["_all"]["cnt"] += 1

        for g, rule in groups.items():
            if in_group(name, rule):
                out[g]["d_sq"] += d2
                out[g]["p_sq"] += p2
                out[g]["cnt"] += 1
                break

    eps = 1e-12
    
    for g, s in out.items():
        if s["cnt"] == 0:
            continue
        r = (s["d_sq"] ** 0.5) / ((s["p_sq"] ** 0.5) + eps)
        writer.add_scalar(f"UpdateRatio/{g}", r, step)

    
    if "queries" in out and "backbone" in out:
        rq = (out["queries"]["d_sq"] ** 0.5) / ((out["queries"]["p_sq"] ** 0.5) + eps)
        rb = (out["backbone"]["d_sq"] ** 0.5) / ((out["backbone"]["p_sq"] ** 0.5) + eps)
        writer.add_scalar("UpdateRatio/queries_over_backbone", rq / (rb + eps), step)


@torch.no_grad()
def log_batch_debug(
    debug_file_path: str,
    epoch: int,
    it: int,
    height_output: dict,
    height: torch.Tensor,
    valid_mask: torch.Tensor,
    opt,
    loss_fn,   
):
    """

      loss = sum(loss_map * w) / sum(w)



      - MAE_in/MAE_bd/MAE_bg


      - pred_mean/pred_std
    """

    pred_h = height_output["height"].detach()
    gt_h = height.detach()
    vmask = (valid_mask > 0.5)

    if not vmask.any():
        return

    # zones
    fp_mask = (gt_h > float(opt.fp_thresh)).float()
    inner, bd, bg = compute_fp_zones(fp_mask)

    m_in = vmask & (inner > 0.5)
    m_bd = vmask & (bd > 0.5)
    m_bg = vmask & (bg > 0.5)

    # enforce mutual exclusive (safer)
    m_bd = m_bd & (~m_in)
    m_bg = m_bg & (~m_in) & (~m_bd)

    n_valid = float(vmask.sum().item())
    n_in = float(m_in.sum().item())
    n_bd = float(m_bd.sum().item())
    n_bg = float(m_bg.sum().item())

    r_in = n_in / (n_valid + 1e-6)
    r_bd = n_bd / (n_valid + 1e-6)
    r_bg = n_bg / (n_valid + 1e-6)

    # loss map consistent with RegressionLoss.loss_fn (use abs for l1, or call loss_fn.loss_fn)
    
    loss_map = loss_fn.loss_fn(pred_h, gt_h)  # [B,1,H,W]
    abs_err = (pred_h - gt_h).abs()

    
    w = vmask.float()
    if getattr(opt, "use_fp_weight", False):
        lam = float(loss_fn.get_lambda())
        fp_weight = (
            float(loss_fn.inner_w) * inner +
            float(loss_fn.boundary_w) * lam * bd +
            float(loss_fn.background_w) * lam * bg
        )
        w = w * fp_weight

    # zone weight maps
    w_in = w * m_in.float()
    w_bd = w * m_bd.float()
    w_bg = w * m_bg.float()

    w_sum_all = float(w.sum().item()) + 1e-6
    w_sum_in = float(w_in.sum().item())
    w_sum_bd = float(w_bd.sum().item())
    w_sum_bg = float(w_bg.sum().item())

    wr_in = w_sum_in / w_sum_all
    wr_bd = w_sum_bd / w_sum_all
    wr_bg = w_sum_bg / w_sum_all

    # numerator contributions: num = sum(loss_map * w)
    num_all = float((loss_map * w).sum().item()) + 1e-6
    num_in = float((loss_map * w_in).sum().item())
    num_bd = float((loss_map * w_bd).sum().item())
    num_bg = float((loss_map * w_bg).sum().item())

    nr_in = num_in / num_all
    nr_bd = num_bd / num_all
    nr_bg = num_bg / num_all

    # plain MAE per zone (not weighted)
    def safe_mean(x, m):
        return float(x[m].mean().item()) if m.any() else 0.0

    mae_in = safe_mean(abs_err, m_in)
    mae_bd = safe_mean(abs_err, m_bd)
    mae_bg = safe_mean(abs_err, m_bg)

    # weighted MAE per zone (optional, helps intuition)
    def safe_wmean(x, w_zone):
        denom = float(w_zone.sum().item())
        if denom <= 1e-6:
            return 0.0
        return float((x * w_zone).sum().item() / (denom + 1e-6))

    wmae_in = safe_wmean(abs_err, w_in)
    wmae_bd = safe_wmean(abs_err, w_bd)
    wmae_bg = safe_wmean(abs_err, w_bg)

    pred_mean = float(pred_h[vmask].mean().item())
    pred_std = float(pred_h[vmask].std().item())

    # write
    need_header = (not os.path.exists(debug_file_path)) or (os.path.getsize(debug_file_path) == 0)
    with open(debug_file_path, "a") as df:
        if need_header:
            df.write(
                "epoch | it   | r_in  r_bd  r_bg  | "
                "wr_in wr_bd wr_bg | nr_in nr_bd nr_bg | "
                "pred_mean pred_std | "
                "mae_in mae_bd mae_bg | "
                "wmae_in wmae_bd wmae_bg | "
                "w_in w_bd w_bg lam\n"
            )
        df.write(
            f"{epoch:<5d} | {it:<4d} | "
            f"{r_in:>5.3f} {r_bd:>5.3f} {r_bg:>5.3f} | "
            f"{wr_in:>5.3f} {wr_bd:>5.3f} {wr_bg:>5.3f} | "
            f"{nr_in:>5.3f} {nr_bd:>5.3f} {nr_bg:>5.3f} | "
            f"{pred_mean:>8.3f} {pred_std:>8.3f} | "
            f"{mae_in:>6.3f} {mae_bd:>6.3f} {mae_bg:>6.3f} | "
            f"{wmae_in:>6.3f} {wmae_bd:>6.3f} {wmae_bg:>6.3f} | "
            f"{float(loss_fn.inner_w):>4.2f} {float(loss_fn.boundary_w):>4.2f} {float(loss_fn.background_w):>4.2f} {float(loss_fn.get_lambda()):>4.2f}\n"
        )


def log_gradient_norms(model, writer, step, prefix='Total_L2_Norm'):

    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            
            total_norm += p.grad.data.norm(2).item() ** 2
            
            # writer.add_histogram(f'gradients/{p.name}', p.grad, step)
    total_norm = total_norm ** 0.5
    writer.add_scalar('Gradient/'+prefix, total_norm, step)


@torch.no_grad()
def log_module_grad_norms(
    model: torch.nn.Module,
    writer,
    step: int,
    tag_prefix: str = "Gradient/ByModule",
    groups: Optional[Dict[str, List[str]]] = None,
    exclude_keys: Optional[List[str]] = None,
    log_ratio: bool = False,
    log_counts: bool = False,
    log_max_param: bool = False,
):
    """






    Args:





    """
    if groups is None:
        groups = {
            "encoder": ["encoder."],
            "prob_fpn": ["prob_fpn."],
            "fp_fpn": ["fp_fpn."],
            "fp_gate": ["fp_gate."],
            "fp_out": ["fp_out."],

            "detr_decoder": ["decoder.blocks."],  
            "ms_query_head": ["ms_query_head."],  

            "mem_proj": ["mem_proj."],
            "level_embed": ["level_embed."],
            "queries": ["query_feat", "query_pe"],

            "height_heads": ["width_mlp.", "mask_mlp.", "mask_feat_proj.", "logit_scale"],
            "refine": ["height_refine."],

            "others": [],
        }

    if exclude_keys is None:
        exclude_keys = []

    
    stats = {
        g: {"sq": 0.0, "n": 0, "none": 0, "max": 0.0, "nonfinite": 0}
        for g in groups.keys()
    }
    total_sq = 0.0
    nonfinite_cnt = 0

    def match_group(param_name: str) -> str:
        for gname, keys in groups.items():
            if gname == "others":
                continue
            for k in keys:
                if k and (k in param_name):
                    return gname
        return "others"

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in exclude_keys):
            continue

        gname = match_group(name)
        stats[gname]["n"] += 1

        if p.grad is None:
            stats[gname]["none"] += 1
            continue

        g = p.grad.detach()
        if not torch.isfinite(g).all():
            nonfinite_cnt += 1
            stats[gname]["nonfinite"] += 1
            g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

        gn = g.float().norm(2).item()
        sq = gn * gn

        total_sq += sq
        stats[gname]["sq"] += sq
        if gn > stats[gname]["max"]:
            stats[gname]["max"] = gn

    total_norm = (total_sq ** 0.5)
    writer.add_scalar(f"{tag_prefix}/total_norm", total_norm, step)
    if log_counts:
        writer.add_scalar(f"{tag_prefix}/nonfinite_grad_tensors", nonfinite_cnt, step)

    denom = total_norm + 1e-12
    for gname, d in stats.items():
        gnorm = (d["sq"] ** 0.5)
        writer.add_scalar(f"{tag_prefix}/{gname}_norm", gnorm, step)

        if log_ratio:
            writer.add_scalar(f"{tag_prefix}/{gname}_ratio", gnorm / denom, step)

        if log_max_param:
            writer.add_scalar(f"{tag_prefix}/{gname}_max_param_norm", d["max"], step)

        if log_counts:
            writer.add_scalar(f"{tag_prefix}/{gname}_param_cnt", d["n"], step)
            writer.add_scalar(f"{tag_prefix}/{gname}_none_grad_cnt", d["none"], step)
            writer.add_scalar(f"{tag_prefix}/{gname}_nonfinite_param_grads", d["nonfinite"], step)


def _select_shared_params(model, include_keys=None, exclude_keys=None):
    """



    """
    params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if include_keys and (not any(k in n for k in include_keys)):
            continue
        if exclude_keys and any(k in n for k in exclude_keys):
            continue
        params.append(p)
    return params


@torch.no_grad()
def _grad_l2_norm(grads):
    total_sq = 0.0
    none_cnt = 0
    for g in grads:
        if g is None:
            none_cnt += 1
            continue
        gn = g.detach().float().norm(2).item()
        total_sq += gn * gn
    return total_sq ** 0.5, none_cnt


def log_reg_seg_gradient_norms(
    model,
    writer,
    step: int,
    loss_reg,                 # scalar tensor, has graph
    loss_seg,                 # scalar tensor, has graph (can be None)
    lambda_seg: float = 1.0,  
    target_ratio: float = 0.2,
    include_keys=None,
    exclude_keys=None,
    tag_prefix: str = "GradNorm",
):
    """

      - ||d(loss_reg)/d(theta_shared)||_2
      - ||d(loss_seg)/d(theta_shared)||_2
      - ratio_raw = g_seg / g_reg
      - ratio_weighted = (lambda_seg * g_seg) / g_reg





    """

    if (loss_reg is None) or (not torch.is_tensor(loss_reg)) or (loss_reg.numel() != 1):
        return
    if (loss_reg.grad_fn is None) and (not loss_reg.requires_grad):
        return

    
    if exclude_keys is None:
        exclude_keys = [
            # ---- seg head ----
            "out_fp_logits",
            "fp_refine",

            # ---- bins / mask heads (height head) ----
            "width_mlp",
            "mask_mlp",

            
            "query_feat",
            "query_pe",
            "level_embed",

            
            
            # "mask_feat_proj",
        ]

    params = _select_shared_params(model, include_keys=include_keys, exclude_keys=exclude_keys)
    if len(params) == 0:
        return

    # --- reg grad norm ---
    grads_reg = torch.autograd.grad(
        loss_reg, params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True
    )
    g_reg, none_reg = _grad_l2_norm(grads_reg)
    writer.add_scalar(f"{tag_prefix}/g_reg", g_reg, step)
    writer.add_scalar(f"{tag_prefix}/g_reg_none_cnt", none_reg, step)
    writer.add_scalar(f"{tag_prefix}/loss_reg", float(loss_reg.detach().item()), step)

    # --- seg grad norm ---
    g_seg = 0.0
    none_seg = 0
    if loss_seg is not None and torch.is_tensor(loss_seg) and loss_seg.numel() == 1:
        if (loss_seg.grad_fn is not None) or loss_seg.requires_grad:
            grads_seg = torch.autograd.grad(
                loss_seg, params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )
            g_seg, none_seg = _grad_l2_norm(grads_seg)

    writer.add_scalar(f"{tag_prefix}/g_seg", g_seg, step)
    writer.add_scalar(f"{tag_prefix}/g_seg_none_cnt", none_seg, step)
    if loss_seg is not None and torch.is_tensor(loss_seg):
        writer.add_scalar(f"{tag_prefix}/loss_seg", float(loss_seg.detach().item()), step)

    # --- ratios ---
    eps = 1e-12
    ratio_raw = g_seg / (g_reg + eps)
    ratio_weighted = (float(lambda_seg) * g_seg) / (g_reg + eps)

    writer.add_scalar(f"{tag_prefix}/ratio_raw(g_seg_over_g_reg)", ratio_raw, step)
    writer.add_scalar(f"{tag_prefix}/ratio_weighted(lambda*g_seg_over_g_reg)", ratio_weighted, step)

    # --- suggested lambda_seg to hit target_ratio ---
    # want: (lambda_seg * g_seg) / g_reg ≈ target_ratio  =>  lambda_seg ≈ target_ratio * g_reg / g_seg
    if g_seg > 0:
        lambda_suggest = float(target_ratio) * (g_reg / (g_seg + eps))
        writer.add_scalar(f"{tag_prefix}/lambda_seg_suggest(target={target_ratio})", lambda_suggest, step)


def _infer_city_and_patch_from_ref(ref_path: str):
    ref_path = Path(ref_path)
    try:
        city = ref_path.parents[1].name
        if city == "":
            city = "UnknownCity"
    except Exception:
        city = "UnknownCity"
    patch = ref_path.stem
    return city, patch


def _to_uint16_gray(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=vmin, posinf=vmax, neginf=vmin)
    if vmax <= vmin + 1e-12:
        return np.zeros_like(arr, dtype=np.uint16)
    x = (arr - vmin) / (vmax - vmin)
    x = np.clip(x, 0.0, 1.0)
    return (x * 65535.0 + 0.5).astype(np.uint16)


def save_gray_png_like_pred_tif(
    data_2d: np.ndarray,
    out_dir: str,
    ref_path: str,
    valid_mask_2d: Optional[np.ndarray] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    percentiles: tuple = (1.0, 99.0),
):
    """

    - data_2d: float32 2D array


    """
    out_dir = Path(out_dir)
    city, patch = _infer_city_and_patch_from_ref(ref_path)
    city_dir = out_dir / city
    city_dir.mkdir(parents=True, exist_ok=True)
    out_path = city_dir / f"{patch}.png"

    arr = np.asarray(data_2d, dtype=np.float32)

    if valid_mask_2d is not None:
        vm = np.asarray(valid_mask_2d, dtype=np.float32) > 0.5
        if vm.any():
            vals = arr[vm]
        else:
            vals = arr.reshape(-1)
    else:
        vals = arr.reshape(-1)

    if vmin is None:
        vmin = float(np.percentile(vals, float(percentiles[0]))) if vals.size > 0 else 0.0
    if vmax is None:
        vmax = float(np.percentile(vals, float(percentiles[1]))) if vals.size > 0 else (vmin + 1.0)

    img_u16 = _to_uint16_gray(arr, vmin=vmin, vmax=vmax)
    Image.fromarray(img_u16, mode="I;16").save(str(out_path))


@torch.no_grad()
def save_intermediate_feature_pngs(
    output_1: Dict[str, torch.Tensor],
    ref_path: str,
    out_root: str,
    valid_mask_1: Optional[torch.Tensor] = None,
    prefix: str = "mid_",
    first_channel_only: bool = True,
    percentiles: tuple = (1.0, 99.0),
    max_items: int = 50,
    
    save_logits_diagnostics: bool = True,
    save_logits_bins: bool = True,
):
    """

      out_root/<tag>/<city>/<patch>.png



    """
    import math
    import numpy as np
    import torch
    import torch.nn.functional as F
    from pathlib import Path

    # --------- helper: resize valid mask to target H,W ----------
    def _mask_to_hw(mask: Optional[torch.Tensor], H: int, W: int) -> Optional[np.ndarray]:
        """
        mask: [1,1,H0,W0] or [1,H0,W0] or [H0,W0]
        return: np.float32 [H,W] in {0,1} (actually float mask)
        """
        if mask is None:
            return None

        m = mask
        if torch.is_tensor(m):
            if m.ndim == 2:
                m = m.unsqueeze(0).unsqueeze(0)      # [1,1,H0,W0]
            elif m.ndim == 3:
                m = m.unsqueeze(1)                   # [1,1,H0,W0]
            elif m.ndim == 4:
                pass
            else:
                return None

            m = m.float()
            if (m.shape[-2] != H) or (m.shape[-1] != W):
                
                m = F.interpolate(m, size=(H, W), mode="nearest")
            m2d = (m[0, 0] > 0.5).detach().cpu().numpy().astype(np.float32)
            return m2d

        
        m = np.asarray(m).astype(np.float32)
        if m.ndim == 2 and (m.shape[0] != H or m.shape[1] != W):
            
            ys = (np.linspace(0, m.shape[0] - 1, H)).astype(np.int32)
            xs = (np.linspace(0, m.shape[1] - 1, W)).astype(np.int32)
            m = m[ys][:, xs]
        return (m > 0.5).astype(np.float32)

    # --------- helper: save wrapper ----------
    def _save(tag: str, arr2d: np.ndarray, vm2d: Optional[np.ndarray], vmin=None, vmax=None, pct=percentiles):
        save_gray_png_like_pred_tif(
            data_2d=arr2d.astype(np.float32),
            out_dir=str(Path(out_root) / tag),
            ref_path=ref_path,
            valid_mask_2d=vm2d,
            vmin=vmin,
            vmax=vmax,
            percentiles=pct,
        )

    cnt = 0
    for k, v in output_1.items():
        if cnt >= max_items:
            break
        if not isinstance(k, str) or not k.startswith(prefix):
            continue
        if not torch.is_tensor(v):
            continue
        if v.ndim != 4 or v.shape[0] != 1:
            continue  

        name = k[len(prefix):]
        C = int(v.shape[1])
        H, W = int(v.shape[2]), int(v.shape[3])

        
        vm2d = _mask_to_hw(valid_mask_1, H, W)

        
        if ("w_build" in name.lower()) and (v.shape[2] == 1) and (v.shape[3] == 1) and (v.shape[1] > 1):
            # v: [1,K,1,1] -> [K]
            vec = v[0, :, 0, 0].detach().float().cpu().numpy()
            save_bin_vector_txt(
                vec=vec,
                out_dir=str(Path(out_root) / name),   # out_root/w_build/<city>/<patch>.txt
                ref_path=ref_path,
                title=name
            )
            cnt += 1
            continue

        # --------------------------
        
        # --------------------------
        is_logits_like = ("logit" in name.lower() or name.lower().endswith("logits"))
        if save_logits_diagnostics and is_logits_like and C >= 4:
            logits = v[0].float()  # [K,H,W]
            logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
            prob = torch.softmax(logits, dim=0)  # [K,H,W]

            # argmax index
            argmax_idx = torch.argmax(prob, dim=0).detach().cpu().numpy().astype(np.float32)
            _save(f"{name}_argmax_idx", argmax_idx, vm2d, vmin=0.0, vmax=float(C - 1), pct=(0.0, 100.0))

            # top1 prob
            # top1 = torch.max(prob, dim=0).values.detach().cpu().numpy().astype(np.float32)
            # _save(f"{name}_top1_prob", top1, vm2d, vmin=0.0, vmax=1.0, pct=(0.0, 100.0))

            # entropy in [0, logK]
            # ent = -(prob * (prob.clamp_min(1e-6).log())).sum(dim=0)
            # ent = ent.detach().cpu().numpy().astype(np.float32)
            # _save(f"{name}_entropy", ent, vm2d, vmin=0.0, vmax=float(math.log(max(C, 2))), pct=(0.0, 100.0))

            # margin = top1 - top2
            # top2 = torch.topk(prob, k=2, dim=0).values  # [2,H,W]
            # margin = (top2[0] - top2[1]).detach().cpu().numpy().astype(np.float32)
            # _save(f"{name}_margin", margin, vm2d, vmin=0.0, vmax=1.0, pct=(0.0, 100.0))

            # optional: three representative bins prob maps
            if save_logits_bins:
                idxs = [0, C // 2, C - 1]
                for bi in idxs:
                    pb = prob[bi].detach().cpu().numpy().astype(np.float32)
                    _save(f"{name}_prob_bin{bi}", pb, vm2d, vmin=0.0, vmax=1.0, pct=(0.0, 100.0))

            cnt += 1
            continue

        # --------------------------
        
        # --------------------------
        if first_channel_only:
            feat = v[0, 0].detach().float().cpu().numpy()
            _save(name, feat, vm2d, vmin=None, vmax=None, pct=percentiles)
        else:
            feat = v[0, 0].detach().float().cpu().numpy()
            _save(name, feat, vm2d, vmin=None, vmax=None, pct=percentiles)

        cnt += 1


def save_bin_vector_txt(vec, out_dir, ref_path, title="w_build"):
    """



    """
    out_dir = Path(out_dir)
    ref_path = Path(ref_path)

    
    try:
        city = ref_path.parents[1].name
        if city == "":
            city = "UnknownCity"
    except Exception:
        city = "UnknownCity"

    city_dir = out_dir / city
    city_dir.mkdir(parents=True, exist_ok=True)

    patch_name = ref_path.stem
    out_path = city_dir / f"{patch_name}.txt"

    vec = np.asarray(vec, dtype=np.float32).reshape(-1)

    with open(out_path, "w") as f:
        f.write(f"{title} (len={len(vec)}):\n")
        f.write("idx\tvalue\n")
        for i, x in enumerate(vec):
            f.write(f"{i}\t{x:.6f}\n")


def save_pred_tif(data, out_dir, ref_path, mask=None, nodata=-1.0, clip_negative=False):
    """

    """
    out_dir = Path(out_dir)
    ref_path = Path(ref_path)

    with rasterio.open(ref_path) as src:
        profile = src.profile.copy()

    data = np.asarray(data, dtype=np.float32)

    # mask -> nodata
    if mask is not None:
        mask = np.asarray(mask)
        data = data.copy()
        data[mask < 0.5] = np.float32(nodata)

    # only for height maps
    if clip_negative:
        data = np.where(data == np.float32(nodata), data, np.maximum(data, 0).astype(np.float32))

    profile.update(dtype=rasterio.float32, count=1, nodata=np.float32(nodata))

    # ---- infer city name robustly ----
    try:
        city = ref_path.parents[1].name  
        if city == "":
            city = "UnknownCity"
    except Exception:
        city = "UnknownCity"

    city_dir = out_dir / city
    city_dir.mkdir(parents=True, exist_ok=True)

    patch_name = ref_path.stem  # no suffix
    out_path = city_dir / f"{patch_name}.tif"

    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(data, 1)


def save_widths_and_edges(bin_widths, bin_edges, out_dir, ref_path):
    """


    """
    out_dir = Path(out_dir)
    ref_path = Path(ref_path)

    
    try:
        city = ref_path.parents[1].name  
        if city == "":
            city = "UnknownCity"
    except Exception:
        city = "UnknownCity"

    city_dir = out_dir / city
    city_dir.mkdir(parents=True, exist_ok=True)

    patch_name = ref_path.stem  

    
    out_path = city_dir / f"{patch_name}.txt"

    bin_widths = np.asarray(bin_widths, dtype=np.float32)  # bin_widths
    bin_edges = np.asarray(bin_edges, dtype=np.float32)    # bin_edges
    with open(out_path, 'w') as f:
        f.write("bin_widths:\n")
        f.write(" ".join([f"{x:.4f}".ljust(8) for x in bin_widths]) + '\n')
        f.write("\nbin_edges:\n")
        f.write(" ".join([f"{x:.4f}".ljust(8) for x in bin_edges]) + '\n')


def save_pred_metrics(mode, final, patch_list, log_path, use_bins=False):
    use_bins = bool(use_bins and ('bin_acc_pm1' in final) and ('bin_mae' in final))
    if mode in ['reg', 'multi']:
        header = (f"{'MRE':^10} | {'MAE':^10} | {'RMSE':^9} | {'RMSE_all':^9} | {'Sigma_1':^9} | {'Sigma_2':^10} | {'Sigma_3':^10} | "
                  f"{'R2':^12} | {'Precision':^9} | {'IoU':^14} | {'Recall':^13} | {'F1-Score':^8}")
        total_line = (f"{final['MRE']:^10.4f} | {final['MAE']:^10.4f} | {final['RMSE']:^10.4f} | {final.get('RMSE_all', 0.0):^10.4f} | {final['Sigma_1']:^11.4f} | "
                      f"{final['Sigma_2']:^12.4f} | {final['Sigma_3']:^11.4f} | {final['R2']:^10.4f} | {final['Precision']:^9.4f} | "
                      f"{final['IoU']:^13.4f} | {final['Recall']:^12.4f} | {final['F1-Score']:^9.4f}")
        if use_bins:
            header += f" | {'BinAcc±1':^8} | {'BinMAE':^9} | {'EBinMAE':^9}"
            total_line += f" | {final['bin_acc_pm1']:^11.4f} | {final['bin_mae']:^10.4f} | {final.get('ebin_mae', 0.0):^10.4f}"
        header += f" | {'Num_Pixels':^14} | {'Num_Pixels_FP':^14} | {'Patch':^48}"
        total_line += f" | {'':^21} | {'':^21} | {'Total':^50}"
        patch_list.sort(key=lambda item: item['RMSE'])
    else:
        header = f"{'Precision':^9} | {'IoU':^13} | {'Recall':^11} | {'F1-Score':^8} | {'Num_Pixels':^15} | {'Patch':^48}"
        total_line = (f"{final['Precision']:^10.4f} | {final['IoU']:^10.4f} | {final['Recall']:^10.4f} | {final['F1-Score']:^10.4f} | "
                      f"{'':^25} | {'Total':^50}")
        patch_list.sort(key=lambda item: item['F1-Score'], reverse=True)

    lines = []
    lines.append(header)
    lines.append(f"{'Total Metrics:':<50}")
    lines.append(total_line)
    lines.append(f"{'Patch Metrics:':<50}")

    for patch in patch_list:
        if patch['Num_Pixels_FP'] == 0:
            continue        

        height_path = patch['Label_Path']
        patch_name = os.path.splitext(os.path.basename(height_path))[0].split('_')[0]
        patch_dir = height_path.split('/')[-3]
        patch_info = patch_dir + '_' + patch_name

        if mode in ['reg', 'multi']:
            patch_line = (
                f"{patch['MRE']:^10.4f} | {patch['MAE']:^10.4f} | {patch['RMSE']:^10.4f} | {patch.get('RMSE_all', 0.0):^10.4f} | "
                f"{patch['Sigma_1']:^11.4f} | {patch['Sigma_2']:^12.4f} | {patch['Sigma_3']:^11.4f} | "
                f"{patch['R2']:^10.4f} | {patch['Precision']:^9.4f} | {patch['IoU']:^13.4f} | "
                f"{patch['Recall']:^12.4f} | {patch['F1-Score']:^9.4f}"
            )
            if use_bins:
                patch_line += f" | {patch['bin_acc_pm1']:^11.4f} | {patch['bin_mae']:^10.4f} | {patch.get('ebin_mae', 0.0):^10.4f}"
            patch_line += f" | {patch['Num_Pixels']:^18} | {patch['Num_Pixels_FP']:^19} | {patch_info:<30}"
        else:
            patch_line = (
                f"{patch['Precision']:^10.4f} | {patch['IoU']:^10.4f} | {patch['Recall']:^10.4f} | "
                f"{patch['F1-Score']:^10.4f} | {patch['Num_Pixels']:^19} | {patch_info:<30}"
            )

        lines.append(patch_line)

    with open(log_path, 'w') as f:
        f.write("\n".join(lines) + "\n")


def save_pred_png_height(height_2d, out_dir, ref_path, valid_mask_2d=None, vmin=0.0, vmax=50.0, bg_max=2.0,
                         
                         seg_edges=(2, 5, 10, 15, 20, 25, 30, 40, 50),
                         
                         bg_color="#f2f2f2", over_color="#7a5cff",
                         
                         seg_colors=(
                                 "#f2f2f2",  
                                 "#b9ddff",  
                                 "#6fa8ff",  
                                 "#3cc7c5",  
                                 "#4dbb6a",  
                                 "#9fd56b",  
                                 "#f1d66a",  
                                 "#f08a4b",  
                                 "#b5162a"   
                         ),
                         invalid_color=(1.0, 1.0, 1.0, 0.0),  
                         clip_negative=True,
                         upscale=4,  
                         
                         colorbar_out_path=None,            # e.g. ".../pred_png_height/_colorbar.png"
                         colorbar_orientation="horizontal", # "horizontal" or "vertical"
                         colorbar_figsize=(6.0, 0.8),       
                         colorbar_dpi=600,
                         colorbar_tick_values=(0, 2, 5, 10, 15, 20, 25, 30, 40, 50),
                         colorbar_tick_labels=("0", "2", "5", "10", "15", "20", "25", "30", "40", "50"), 
                         colorbar_label="Height (m)",
                         colorbar_over_text=">50",
                         colorbar_font_size=10):
    """









    """
    from pathlib import Path
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    import imageio.v2 as imageio

    out_dir = Path(out_dir)
    ref_path = Path(ref_path)

    h = np.asarray(height_2d, dtype=np.float32)
    if clip_negative:
        h = np.maximum(h, 0.0)

    if valid_mask_2d is not None:
        vm = np.asarray(valid_mask_2d, dtype=np.float32)
        valid = (vm > 0.5)
    else:
        valid = np.ones_like(h, dtype=bool)

    # --------- sanity checks ----------
    seg_edges = tuple(float(x) for x in seg_edges)
    if not (len(seg_edges) >= 2 and abs(seg_edges[-1] - float(vmax)) < 1e-6):
        raise ValueError(f"seg_edges must end at vmax, e.g. (..., {vmax}); got {seg_edges}")
    if any(seg_edges[i] >= seg_edges[i + 1] for i in range(len(seg_edges) - 1)):
        raise ValueError(f"seg_edges must be strictly increasing; got {seg_edges}")
    if abs(seg_edges[0] - float(bg_max)) > 1e-6:
        raise ValueError(f"seg_edges[0] must equal bg_max={bg_max}; got {seg_edges[0]}")
    if len(seg_colors) != len(seg_edges):
        raise ValueError(f"seg_colors length must equal seg_edges length: {len(seg_colors)} vs {len(seg_edges)}")

    # --------- build custom "segmented-but-continuous" colormap ----------
    
    knots_h = [float(vmin), float(bg_max)] + list(seg_edges[1:])  
    knots_pos = [(x - float(vmin)) / max(float(vmax) - float(vmin), 1e-6) for x in knots_h]
    knots_pos = [float(np.clip(p, 0.0, 1.0)) for p in knots_pos]

    knot_colors = [mcolors.to_rgba(bg_color), mcolors.to_rgba(bg_color)]
    for c in seg_colors[1:]:  
        knot_colors.append(mcolors.to_rgba(c))

    pos_color = list(zip(knots_pos, knot_colors))
    cmap = mcolors.LinearSegmentedColormap.from_list("height_seg_continuous", pos_color, N=256)
    
    cmap.set_over(mcolors.to_rgba(over_color))

    
    norm = mcolors.Normalize(vmin=float(vmin), vmax=float(vmax), clip=False)

    # --------- save colorbar once if requested ----------
    if colorbar_out_path is not None:
        cb_path = Path(colorbar_out_path)
        if not cb_path.exists():
            cb_path.parent.mkdir(parents=True, exist_ok=True)

            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])

            fig = plt.figure(figsize=colorbar_figsize, dpi=colorbar_dpi)

            if str(colorbar_orientation).lower().startswith("h"):
                
                ax = fig.add_axes([0.06, 0.35, 0.88, 0.35])
                cbar = plt.colorbar(sm, cax=ax, orientation="horizontal", extend="max")
            else:
                
                ax = fig.add_axes([0.25, 0.06, 0.25, 0.88])
                cbar = plt.colorbar(sm, cax=ax, orientation="vertical", extend="max")

            cbar.ax.tick_params(labelsize=colorbar_font_size)

            tick_values = list(colorbar_tick_values)
            cbar.set_ticks(tick_values)

            if colorbar_tick_labels is None:
                tick_labels = [str(t) for t in tick_values]
            else:
                tick_labels = list(colorbar_tick_labels)
            cbar.set_ticklabels(tick_labels)

            if colorbar_label is not None and len(str(colorbar_label)) > 0:
                cbar.set_label(colorbar_label, fontsize=colorbar_font_size)

            
            if colorbar_over_text is not None and len(str(colorbar_over_text)) > 0:
                if str(colorbar_orientation).lower().startswith("h"):
                    cbar.ax.text(1.06, 0.5, str(colorbar_over_text),
                                 va="center", ha="left",
                                 transform=cbar.ax.transAxes,
                                 fontsize=colorbar_font_size)
                else:
                    cbar.ax.text(0.5, 1.02, str(colorbar_over_text),
                                 va="bottom", ha="center",
                                 transform=cbar.ax.transAxes,
                                 fontsize=colorbar_font_size)

            fig.savefig(str(cb_path), transparent=True, bbox_inches="tight", pad_inches=0.06)
            plt.close(fig)

    # --------- map to RGBA for the image ----------
    rgba = cmap(norm(h))  # float RGBA in [0,1]

    
    over_rgba = np.array(mcolors.to_rgba(over_color), dtype=np.float32)
    rgba[h > float(vmax)] = over_rgba

    # invalid -> transparent by default
    inv_rgba = np.array(invalid_color, dtype=np.float32)
    rgba[~valid] = inv_rgba

    rgba_u8 = (np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8)
    if int(upscale) > 1:
        s = int(upscale)
        rgba_u8 = np.repeat(np.repeat(rgba_u8, s, axis=0), s, axis=1)

    # --------- keep the same folder habit as save_pred_tif ----------
    # out_dir/<city>/<patch>.png
    try:
        city = ref_path.parents[1].name or "UnknownCity"
    except Exception:
        city = "UnknownCity"

    city_dir = out_dir / city
    city_dir.mkdir(parents=True, exist_ok=True)

    patch_name = ref_path.stem
    out_path = city_dir / f"{patch_name}.png"
    imageio.imwrite(str(out_path), rgba_u8)


def save_pred_png_rgb(optical_7ch,
                      out_dir,
                      ref_path,
                      valid_mask_2d=None,
                      band_indices=(3, 2, 1),
                      
                      stretch="percentile",   # "percentile" | "minmax" | "none"
                      p_low=2.0,
                      p_high=98.0,
                      gamma=1.0,
                      invalid_rgb=(235, 235, 235),  
                      clip_to_unit=True,
                      upscale=4):
    """







    stretch:




    upscale:

    """
    from pathlib import Path
    import numpy as np
    import imageio.v2 as imageio

    # torch -> numpy
    if hasattr(optical_7ch, "detach"):
        x = optical_7ch.detach().cpu().numpy()
    else:
        x = np.asarray(optical_7ch)

    x = x.astype(np.float32)

    
    if x.ndim != 3:
        raise ValueError(f"optical_7ch must be 3D, got shape={x.shape}")
    if x.shape[0] in (7, 6, 8):  # CHW
        chw = x
    else:  # HWC
        chw = np.transpose(x, (2, 0, 1))

    C, H, W = chw.shape
    # 1-based -> 0-based
    idx0 = [int(b) - 1 for b in band_indices]
    if any(b < 0 or b >= C for b in idx0):
        raise ValueError(f"band_indices {band_indices} out of range for C={C}")

    rgb = np.stack([chw[idx0[0]], chw[idx0[1]], chw[idx0[2]]], axis=-1)  # [H,W,3]

    # valid mask
    if valid_mask_2d is not None:
        vm = np.asarray(valid_mask_2d).astype(np.float32)
        valid = vm > 0.5
    else:
        valid = np.ones((H, W), dtype=bool)

    
    if stretch not in ("percentile", "minmax", "none"):
        raise ValueError(f"stretch must be percentile|minmax|none, got {stretch}")

    rgb01 = np.zeros_like(rgb, dtype=np.float32)

    for c in range(3):
        ch = rgb[..., c]
        ch_valid = ch[valid] if valid.any() else ch.reshape(-1)

        if stretch == "none":
            ch_n = ch
        elif stretch == "minmax":
            lo = float(np.min(ch_valid)) if ch_valid.size else 0.0
            hi = float(np.max(ch_valid)) if ch_valid.size else 1.0
            ch_n = (ch - lo) / max(hi - lo, 1e-6)
        else:  # percentile
            lo = float(np.percentile(ch_valid, p_low)) if ch_valid.size else 0.0
            hi = float(np.percentile(ch_valid, p_high)) if ch_valid.size else 1.0
            ch_n = (ch - lo) / max(hi - lo, 1e-6)

        if clip_to_unit:
            ch_n = np.clip(ch_n, 0.0, 1.0)

        if gamma is not None and float(gamma) != 1.0:
            g = float(gamma)
            ch_n = np.power(np.clip(ch_n, 0.0, 1.0), 1.0 / g)

        rgb01[..., c] = ch_n

    rgb_u8 = (rgb01 * 255.0 + 0.5).astype(np.uint8)

    
    if valid_mask_2d is not None:
        inv = ~valid
        if inv.any():
            rgb_u8[inv] = np.array(invalid_rgb, dtype=np.uint8)

    
    if int(upscale) > 1:
        s = int(upscale)
        rgb_u8 = np.repeat(np.repeat(rgb_u8, s, axis=0), s, axis=1)

    
    out_dir = Path(out_dir)
    ref_path = Path(ref_path)
    try:
        city = ref_path.parents[1].name or "UnknownCity"
    except Exception:
        city = "UnknownCity"
    city_dir = out_dir / city
    city_dir.mkdir(parents=True, exist_ok=True)

    patch_name = ref_path.stem
    out_path = city_dir / f"{patch_name}.png"
    imageio.imwrite(str(out_path), rgb_u8)


if __name__ == "__main__":

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import pandas as pd

    exp_dir = r"E:\HeightMap\DL_Results\RestBinsFormer_test"
    event_path = os.path.join(exp_dir, "events.out.tfevents.1768886319.g0005.2445640.0")
    ea = EventAccumulator(event_path)
    ea.Reload()

    tags = ea.Tags()["scalars"]
    print("Scalar tags:", tags)


    def dump(tag):
        evs = ea.Scalars(tag)
        return pd.DataFrame([(e.step, e.value) for e in evs], columns=["step", "value"])


    need = [
        "GradNorm/g_reg",
        "GradNorm/g_seg",
        "GradNorm/ratio_weighted(lambda*g_seg_over_g_reg)",
        "GradNorm/lambda_seg_suggest(target=0.2)",
        "GradNorm/loss_reg",
        "GradNorm/loss_seg",
    ]
    dfs = []
    for t in need:
        if t in tags:
            d = dump(t)
            d["tag"] = t
            dfs.append(d)

    out = pd.concat(dfs, ignore_index=True)
    out.to_csv(os.path.join(exp_dir, "gradnorm_export.csv"), index=False)
    print(out.groupby("tag")["value"].describe())
