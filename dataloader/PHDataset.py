import os
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


def read_tif(patch_path):
    tif = rasterio.open(patch_path)
    tif_array = tif.read().astype(np.float32)
    return torch.tensor(tif_array)


def get_paths(in_dir, split):
    path_file = os.path.join(in_dir, split + '.txt')
    path_list = []
    with open(path_file) as f:
        for line in f:
            line = line.strip()
            path_list.append(os.path.join(in_dir, line))
    return path_list


def build_city2region():
    region_dict = {
        'EU': ['Barcelona', 'London', 'Madrid', 'Vienna'],
        'NA': ['Boston', 'LasVegas', 'LosAngeles', 'Sacramento', 'SaltLakeCity'],
        'AU': ['Adelaide', 'Melbourne', 'Perth'],
        'AS': ['Ankara', 'Beijing', 'Liuzhou', 'Osaka', 'Tianjin'],
        'SA': ['Barranquilla', 'RiodeJaneiro', 'Santiago', 'SaoPaulo'],
        'AF': ['Alexandria', 'CapeTown', 'Constantine', 'Ouargla', 'Tunis']
    }

    city2region = {}
    for region, cities in region_dict.items():
        for city in cities:
            city2region[city] = region
    strong_cities = {city for city, region in city2region.items() if region in ['EU', 'NA', 'AU']}
    return strong_cities


class PHDataset(Dataset):
    def __init__(self, opt, split='train'):
        self.path_list = get_paths(opt.data_dir, split)
        self.mode = opt.mode
        self.norm_path = opt.norm_path
        self.fp_thresh = opt.fp_thresh

        if opt.norm_path is not None:
            stats = np.load(opt.norm_path)
            self.OPT_MIN_GLOBAL = float(stats['opt_min'])
            self.OPT_MAX_GLOBAL = float(stats['opt_max'])
            self.NODATA_VALUE = float(stats['nodata_value'])
            self.OPT_RANGE = self.OPT_MAX_GLOBAL - self.OPT_MIN_GLOBAL
            if self.OPT_RANGE == 0:
                self.OPT_RANGE = 1e-6

    def __getitem__(self, index):
        optical_path = self.path_list[index]
        height_path = self.path_list[index].replace('optical_patches', 'height_patches')
        optical = read_tif(optical_path)
        height = read_tif(height_path)

        optical_valid_mask = torch.ones_like(optical, dtype=torch.bool)
        height_valid_mask = torch.ones_like(height, dtype=torch.bool)

        if self.norm_path is not None:
            optical_valid_mask = optical != self.NODATA_VALUE
            height_valid_mask = height != self.NODATA_VALUE
            optical = (optical - self.OPT_MIN_GLOBAL) / self.OPT_RANGE
            optical[~optical_valid_mask] = torch.tensor(0, dtype=optical.dtype)
            height[~height_valid_mask] = torch.tensor(0, dtype=height.dtype)

        if height.ndim == 2:
            height = height[None, :, :]
        if height_valid_mask.ndim == 2:
            height_valid_mask = height_valid_mask[None, :, :]

        if self.mode == 'seg':
            height = (height > self.fp_thresh).float()

        return {
            'optical': optical,
            'height': height,
            'optical_valid_mask': optical_valid_mask[0:1, :, :],
            'height_valid_mask': height_valid_mask,
            'height_path': height_path,
        }

    def __len__(self):
        return len(self.path_list)
