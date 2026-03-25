import gc
import os
import time
from typing import Dict

import torch

# Avoid severe CPU slowdowns caused by subnormal values in decoder features.
torch.set_flush_denormal(True)
from torch.amp import autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataloader.PHDataset import PHDataset, build_city2region
from models import create_model
from options.test_options import TestOptions
from utils.metrics import MetricsBundle
from utils.writers import save_gray_png_like_pred_tif, save_intermediate_feature_pngs, save_pred_metrics, save_pred_png_height, save_pred_png_rgb, save_pred_tif, save_widths_and_edges


def cast_pred_for_metrics(pred: Dict[str, torch.Tensor]):
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


if __name__ == '__main__':
    start_time = time.time()
    opt = TestOptions().parse()
    amp_dtype = torch.float16
    output_dir = os.path.join(opt.save_dir, f'test_{opt.ckpt_name}')
    os.makedirs(output_dir, exist_ok=True)

    test_set = PHDataset(opt, split='test')
    if opt.max_test_size is not None and len(test_set) > opt.max_test_size:
        test_set = Subset(test_set, list(range(opt.max_test_size)))
    test_loader = DataLoader(test_set, shuffle=False, num_workers=opt.n_workers, batch_size=opt.batch_size, drop_last=False)
    print(f'{len(test_set)} patches for testing')

    strong_list = build_city2region()

    model = create_model(opt)
    model_path = os.path.join(opt.save_dir, f'{opt.ckpt_name}.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found: {model_path}')
    print(f'Loading model from: {model_path}')
    checkpoint = torch.load(model_path, map_location=opt.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(opt.device)
    model.eval()

    metrics_filepath = os.path.join(output_dir, 'test_metrics_log.txt')
    per_patch_metrics_log = []
    if opt.separate:
        metrics_ENA_filepath = os.path.join(output_dir, 'test_metrics_ENA_log.txt')
        metrics_ASA_filepath = os.path.join(output_dir, 'test_metrics_ASA_log.txt')
        per_patch_metrics_ENA_log = []
        per_patch_metrics_ASA_log = []

    meter_all = MetricsBundle(mode=opt.mode, fp_thresh=opt.fp_thresh, seg_threshold=0.5, enable_bin_class=bool(opt.use_bins), seg_from_logits=True, fp_temp=opt.fp_temp, seg_source='height', seg_return='binary')
    meter_all.reset()
    patch_meter = MetricsBundle(mode=opt.mode, fp_thresh=opt.fp_thresh, seg_threshold=0.5, enable_bin_class=bool(opt.use_bins), seg_from_logits=True, fp_temp=opt.fp_temp, seg_source='height', seg_return='binary')
    if opt.separate:
        meter_ENA = MetricsBundle(opt.mode, opt.fp_thresh, 0.5, enable_bin_class=bool(opt.use_bins), seg_from_logits=True, fp_temp=opt.fp_temp, seg_source='height', seg_return='binary')
        meter_ASA = MetricsBundle(opt.mode, opt.fp_thresh, 0.5, enable_bin_class=bool(opt.use_bins), seg_from_logits=True, fp_temp=opt.fp_temp, seg_source='height', seg_return='binary')
        meter_ENA.reset()
        meter_ASA.reset()

    with torch.no_grad():
        for test_data in tqdm(test_loader, desc='Testing'):
            optical = test_data['optical'].to(opt.device)
            height = test_data['height'].to(opt.device)
            opt_valid_mask = test_data['optical_valid_mask'].to(opt.device)
            ht_valid_mask = test_data['height_valid_mask'].to(opt.device)
            valid_mask = opt_valid_mask * ht_valid_mask
            patch_height_path = test_data['height_path']

            with autocast(device_type='cuda', enabled=opt.use_amp, dtype=amp_dtype):
                height_output = model(optical)
            height_output = cast_pred_for_metrics(height_output)

            meter_all.update(height_output, height, valid_mask)
            if opt.separate:
                is_ENA = torch.tensor([any(city in p for city in strong_list) for p in patch_height_path], device=opt.device, dtype=torch.bool)
                if is_ENA.any():
                    out_ENA = {k: v[is_ENA] for k, v in height_output.items() if torch.is_tensor(v)}
                    meter_ENA.update(out_ENA, height[is_ENA], valid_mask[is_ENA])
                if (~is_ENA).any():
                    out_ASA = {k: v[~is_ENA] for k, v in height_output.items() if torch.is_tensor(v)}
                    meter_ASA.update(out_ASA, height[~is_ENA], valid_mask[~is_ENA])

            if opt.mode in ['reg', 'multi']:
                pred_h = MetricsBundle.pred_height_from_output(height_output)
                height_output_save = pred_h
            else:
                pred_h = None
                height_output_save = torch.sigmoid(height_output['fp_logits'])

            bin_pred = None
            if 'height_bin_logits' in height_output:
                bin_prob = torch.softmax(height_output['height_bin_logits'], dim=1)
                bin_pred = torch.argmax(bin_prob, dim=1).float()

            for j in range(height.shape[0]):
                patch_meter.reset()
                out_j = {k: v[j:j + 1] for k, v in height_output.items() if torch.is_tensor(v)}
                patch_meter.update(out_j, height[j:j + 1], valid_mask[j:j + 1])
                per_patch_metrics = patch_meter.compute()
                per_patch_metrics['Label_Path'] = patch_height_path[j]
                per_patch_metrics_log.append(per_patch_metrics)

                if opt.separate:
                    city_in_path = next((city for city in strong_list if city in patch_height_path[j]), None)
                    if city_in_path:
                        per_patch_metrics_ENA_log.append(per_patch_metrics)
                    else:
                        per_patch_metrics_ASA_log.append(per_patch_metrics)

                if opt.save_fig:
                    save_pred_tif(height_output_save[j, 0].cpu().numpy(), os.path.join(output_dir, '_pred_height'), patch_height_path[j], valid_mask[j, 0].cpu().numpy())
                    save_pred_png_height(height_output_save[j, 0].cpu().numpy(), os.path.join(output_dir, '_pred_height_png'), patch_height_path[j], valid_mask[j, 0].cpu().numpy())
                    if 'fp_logits' in height_output:
                        pred_fp_np = torch.sigmoid(height_output['fp_logits'][j, 0]).cpu().numpy()
                        save_pred_tif(pred_fp_np, os.path.join(output_dir, '_pred_fp'), patch_height_path[j], valid_mask[j, 0].cpu().numpy())
                        save_gray_png_like_pred_tif(pred_fp_np, os.path.join(output_dir, '_pred_fp_png'), patch_height_path[j], valid_mask[j, 0].cpu().numpy())
                    if bin_pred is not None:
                        save_pred_tif(bin_pred[j].cpu().numpy(), os.path.join(output_dir, '_pred_bin'), patch_height_path[j], valid_mask[j, 0].cpu().numpy())
                    save_pred_png_rgb(test_data['optical'][j].cpu().numpy(), os.path.join(output_dir, '_rgb'), patch_height_path[j])

                if opt.save_head:
                    if 'height_bin_logits' in height_output:
                        bin_prob_np = torch.softmax(height_output['height_bin_logits'][j], dim=0).cpu().numpy()
                        for bi in range(bin_prob_np.shape[0]):
                            save_pred_tif(bin_prob_np[bi], os.path.join(output_dir, f'_bin_prob_{bi:02d}'), patch_height_path[j], valid_mask[j, 0].cpu().numpy())
                    if 'bin_widths' in height_output and 'bin_edges' in height_output:
                        save_widths_and_edges(height_output['bin_widths'][j].cpu().numpy(), height_output['bin_edges'][j].cpu().numpy(), os.path.join(output_dir, '_widths_and_edges'), patch_height_path[j])

                if opt.save_mid_feats:
                    save_intermediate_feature_pngs(height_output, os.path.join(output_dir, '_mid_feats'), patch_height_path[j], sample_idx=j)

    final_metric_log = meter_all.compute()
    save_pred_metrics(opt.mode, final_metric_log, per_patch_metrics_log, metrics_filepath, opt.use_bins)

    if opt.separate:
        final_metric_ENA_log = meter_ENA.compute()
        final_metric_ASA_log = meter_ASA.compute()
        save_pred_metrics(opt.mode, final_metric_ENA_log, per_patch_metrics_ENA_log, metrics_ENA_filepath, opt.use_bins)
        save_pred_metrics(opt.mode, final_metric_ASA_log, per_patch_metrics_ASA_log, metrics_ASA_filepath, opt.use_bins)

    print(f'\nFinish testing, total time: {(time.time() - start_time) / 60:.2f} min\n')
    gc.collect()
    torch.cuda.empty_cache()
