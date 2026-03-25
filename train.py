import gc
import os
import time
from typing import Dict

import torch

# Avoid severe CPU slowdowns caused by subnormal values in decoder features.
torch.set_flush_denormal(True)
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def close(self):
            pass

from dataloader.PHDataset import PHDataset
from models import create_model, create_scheduler, resume_check
from options.train_options import TrainOptions
from utils.losses import create_loss
from utils.metrics import MetricsBundle


def cast_pred_for_loss(pred: Dict[str, torch.Tensor]):
    if not isinstance(pred, dict):
        return pred
    keys_fp32 = {"height", "height_bin_logits", "bin_edges", "bin_widths", "fp_logits"}
    out = {}
    for k, v in pred.items():
        if torch.is_tensor(v) and v.dtype in (torch.float16, torch.bfloat16) and k in keys_fp32:
            out[k] = v.float()
        else:
            out[k] = v
    return out


def _fmt_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.3f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.3f}M"
    if n >= 1_000:
        return f"{n / 1_000:.3f}K"
    return str(n)


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_tag, best_value, best_epoch):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_tag": best_tag,
        "best_value": float(best_value),
        "best_tag_epoch": int(best_epoch),
        "best_metric": float(best_value),
        "best_epoch": int(best_epoch),
    }
    torch.save(ckpt, path)


if __name__ == "__main__":
    start_time = time.time()
    opt = TrainOptions().parse()

    amp_dtype = torch.float16
    scaler = GradScaler(device="cuda", enabled=opt.use_amp, init_scale=1024)
    if opt.use_amp:
        print(f"Use AMP. autocast dtype = {amp_dtype}")

    train_set = PHDataset(opt, split="train")
    val_set = PHDataset(opt, split="val")
    if opt.max_train_size is not None and len(train_set) > opt.max_train_size:
        train_set = Subset(train_set, list(range(opt.max_train_size)))
    if opt.max_val_size is not None and len(val_set) > opt.max_val_size:
        val_set = Subset(val_set, list(range(opt.max_val_size)))

    train_loader = DataLoader(train_set, shuffle=True, num_workers=opt.n_workers, batch_size=opt.batch_size, drop_last=True)
    val_loader = DataLoader(val_set, shuffle=False, num_workers=opt.n_workers, batch_size=opt.batch_size, drop_last=True)
    print(f"{len(train_set)} patches for training, {len(val_set)} patches for validation")

    model = create_model(opt)
    loss_fn = create_loss(opt)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-6)
    scheduler = create_scheduler(optimizer, opt)
    model, optimizer, scheduler, start_epoch, _, _ = resume_check(model, optimizer, scheduler, opt)

    writer = SummaryWriter(opt.save_dir)
    metric_file_path = os.path.join(opt.save_dir, "train_metrics_log.txt")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with open(metric_file_path, "w") as f:
        f.write("Model Params:\n")
        f.write(f"Model: {opt.model}\n")
        f.write(f"Total params: {total_params} ({_fmt_params(total_params)})\n")
        f.write(f"Trainable params: {trainable_params} ({_fmt_params(trainable_params)})\n\n")

    best_rmse_fp = float("inf") if opt.mode in ["reg", "multi"] else None
    best_rmse_all = float("inf") if opt.mode in ["reg", "multi"] else None
    best_iou = 0.0
    best_val_loss = float("inf")
    best_epoch_val_loss = -1
    no_improve = 0

    for epoch in range(start_epoch, opt.n_epochs + start_epoch):
        epoch_start_time = time.time()
        model.train()
        epoch_train_loss = 0.0
        epoch_loss_h = 0.0
        epoch_loss_seg = 0.0
        epoch_loss_seg_count = 0
        current_lr = optimizer.param_groups[0]["lr"]

        for train_data in tqdm(train_loader, desc=f"Train Epoch {epoch}/{opt.n_epochs}, LR {current_lr:.6f}"):
            optical = train_data["optical"].to(opt.device)
            height = train_data["height"].to(opt.device)
            valid_mask = train_data["optical_valid_mask"].to(opt.device) * train_data["height_valid_mask"].to(opt.device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=opt.use_amp, dtype=amp_dtype):
                height_output = model(optical)

            with autocast(device_type="cuda", enabled=False):
                height_output_fp32 = cast_pred_for_loss(height_output)
                loss_train, stats = loss_fn(height_output_fp32, height.float(), valid_mask.float())

            if opt.use_amp:
                scaler.scale(loss_train).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.grad_clip, norm_type=2)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_train.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.grad_clip, norm_type=2)
                optimizer.step()

            epoch_train_loss += float(loss_train.item())
            if stats.get("loss_h") is not None:
                epoch_loss_h += float(stats["loss_h"].item())
            if stats.get("loss_seg") is not None:
                epoch_loss_seg += float(stats["loss_seg"].item())
                epoch_loss_seg_count += 1

        avg_train_loss = epoch_train_loss / max(1, len(train_loader))
        avg_loss_h = epoch_loss_h / max(1, len(train_loader))
        avg_loss_seg = epoch_loss_seg / max(1, epoch_loss_seg_count) if epoch_loss_seg_count > 0 else None
        writer.add_scalar("Train/Loss", avg_train_loss, epoch)
        writer.add_scalar("Train/Loss_h", avg_loss_h, epoch)
        if avg_loss_seg is not None:
            writer.add_scalar("Train/Loss_seg", avg_loss_seg, epoch)

        model.eval()
        val_meter = MetricsBundle(
            mode=opt.mode,
            fp_thresh=opt.fp_thresh,
            seg_threshold=0.5,
            enable_bin_class=bool(opt.use_bins),
            seg_from_logits=True,
            fp_temp=opt.fp_temp,
            seg_source="height",
            seg_return="binary",
        )
        val_meter.reset()
        val_total_loss = 0.0

        with torch.no_grad():
            for val_data in tqdm(val_loader, desc=f"Val Epoch {epoch}/{opt.n_epochs}"):
                optical_val = val_data["optical"].to(opt.device)
                height_val = val_data["height"].to(opt.device)
                val_valid_mask = val_data["optical_valid_mask"].to(opt.device) * val_data["height_valid_mask"].to(opt.device)
                with autocast(device_type="cuda", enabled=opt.use_amp, dtype=amp_dtype):
                    height_output_val = model(optical_val)
                height_output_val = cast_pred_for_loss(height_output_val)
                val_meter.update(height_output_val, height_val.float(), val_valid_mask.float())
                with autocast(device_type="cuda", enabled=False):
                    loss_val, _ = loss_fn(height_output_val, height_val.float(), val_valid_mask.float())
                val_total_loss += float(loss_val.item())

        avg_val_loss_total = val_total_loss / max(1, len(val_loader))
        writer.add_scalar("Val/Loss_total", avg_val_loss_total, epoch)

        m = val_meter.compute()
        epoch_val_precision = m["Precision"]
        epoch_val_iou = m["IoU"]
        epoch_val_recall = m["Recall"]
        epoch_val_f1 = m["F1-Score"]
        writer.add_scalar("Val/IoU", epoch_val_iou, epoch)
        writer.add_scalar("Val/F1-Score", epoch_val_f1, epoch)

        with open(metric_file_path, "a") as f:
            if opt.mode in ["reg", "multi"]:
                epoch_val_mre = m["MRE"]
                epoch_val_mae = m["MAE"]
                epoch_val_rmse = m["RMSE"]
                epoch_val_rmse_all = m.get("RMSE_all", 0.0)
                writer.add_scalar("Val/RMSE", epoch_val_rmse, epoch)
                writer.add_scalar("Val/MRE", epoch_val_mre, epoch)
                writer.add_scalar("Val/RMSE_all", epoch_val_rmse_all, epoch)
                header = (
                    f"{'Epoch':^8} | {'Train Loss':^10} | {'Val Loss':^10} | {'MRE':^10} | {'MAE':^10} | {'RMSE_FP':^10} | {'RMSE_ALL':^10} | "
                    f"{'Precision':^10} | {'IoU':^10} | {'Recall':^10} | {'F1-Score':^10}"
                )
                line = (
                    f"{epoch:^8} | {avg_train_loss:^10.4f} | {avg_val_loss_total:^10.4f} | {epoch_val_mre:^10.4f} | {epoch_val_mae:^10.4f} | {epoch_val_rmse:^10.4f} | {epoch_val_rmse_all:^10.4f} | "
                    f"{epoch_val_precision:^10.4f} | {epoch_val_iou:^10.4f} | {epoch_val_recall:^10.4f} | {epoch_val_f1:^10.4f}"
                )
                if opt.use_bins and "bin_acc_pm1" in m:
                    header += f" | {'BinAcc±1':^10} | {'BinMAE':^10} | {'EBinMAE':^10}"
                    line += f" | {m['bin_acc_pm1']:^10.4f} | {m['bin_mae']:^10.4f} | {m.get('ebin_mae', 0.0):^10.4f}"
                    writer.add_scalar("Val/EBinMAE", m.get("ebin_mae", 0.0), epoch)
            else:
                header = f"{'Epoch':^8} | {'Train Loss':^10} | {'Val Loss':^10} | {'IoU':^10} | {'F1-Score':^10} | {'Precision':^10} | {'Recall':^10}"
                line = f"{epoch:^8} | {avg_train_loss:^10.4f} | {avg_val_loss_total:^10.4f} | {epoch_val_iou:^10.4f} | {epoch_val_f1:^10.4f} | {epoch_val_precision:^10.4f} | {epoch_val_recall:^10.4f}"
            if epoch == start_epoch:
                f.write(header + "\n")
            f.write(line + "\n")

        if opt.lr_policy in ["step", "cosine", "warmcos"]:
            scheduler.step()
        elif opt.lr_policy == "plateau":
            scheduler.step(m["RMSE"] if opt.mode in ["reg", "multi"] else epoch_val_iou)

        print(f"End of epoch {epoch} / {opt.n_epochs}\tTime Taken: {time.time() - epoch_start_time:.0f} sec")

        if opt.mode in ["reg", "multi"] and m["RMSE"] < best_rmse_fp:
            best_rmse_fp = m["RMSE"]
            save_checkpoint(os.path.join(opt.save_dir, "model_best_rmse_fp.pth"), epoch, model, optimizer, scheduler, "rmse_fp", best_rmse_fp, epoch)
        if opt.mode in ["reg", "multi"] and m.get("RMSE_all", 0.0) < best_rmse_all:
            best_rmse_all = m.get("RMSE_all", 0.0)
            save_checkpoint(os.path.join(opt.save_dir, "model_best_rmse_all.pth"), epoch, model, optimizer, scheduler, "rmse_all", best_rmse_all, epoch)
        if epoch_val_iou > best_iou:
            best_iou = epoch_val_iou
            save_checkpoint(os.path.join(opt.save_dir, "model_best_iou.pth"), epoch, model, optimizer, scheduler, "iou", best_iou, epoch)
        if avg_val_loss_total < best_val_loss:
            best_val_loss = avg_val_loss_total
            best_epoch_val_loss = epoch
            no_improve = 0
            save_checkpoint(os.path.join(opt.save_dir, "model_best_val_loss.pth"), epoch, model, optimizer, scheduler, "val_loss", best_val_loss, best_epoch_val_loss)
        else:
            no_improve += 1

        if epoch % opt.save_freq == 0:
            save_checkpoint(os.path.join(opt.save_dir, f"model_{epoch}.pth"), epoch, model, optimizer, scheduler, "val_loss", best_val_loss, best_epoch_val_loss)

        if opt.early_stop and no_improve >= opt.es_patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    writer.close()
    print(f"\nTraining finished, total time: {(time.time() - start_time) / 60:.2f} min\n")
    gc.collect()
    torch.cuda.empty_cache()
