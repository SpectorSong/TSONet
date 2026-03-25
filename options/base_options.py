import argparse
import os
import torch


class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--name', type=str, default='demo_try', help='experiment name')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids, use -1 for CPU')
        parser.add_argument('--result_dir', type=str, default=r'E:/HeightMap/DL_Results', help='result root directory')

        parser.add_argument('--model', type=str, default='tsonet', choices=['tsonet'])
        parser.add_argument('--in_channels', type=int, default=7)
        parser.add_argument('--out_channels', type=int, default=1)
        parser.add_argument('--mode', type=str, default='multi', choices=['reg', 'seg', 'multi'])
        parser.add_argument('--dim', type=int, default=32)
        parser.add_argument('--hidden_dim', type=int, default=256)

        parser.add_argument('--query_mode', type=str, default='febr', choices=['febr', 'detr'], help='bin-head type used in TSONet ablations')
        parser.add_argument('--mem_levels', type=int, nargs='+', default=[5, 4, 3])
        parser.add_argument('--ms_pool_mode', type=str, default='single', choices=['single', 'ms_late', 'iterative'])
        parser.add_argument('--ms_pool_fuse_learnable', action='store_true')
        parser.add_argument('--pool_iter_anchor_mix', action='store_true')
        parser.add_argument('--pool_iter_anchor_mix_mode', type=str, default='per_stage', choices=['scalar', 'per_stage'])
        parser.add_argument('--pool_iter_anchor_mix_init', type=float, default=-2.0)
        parser.add_argument('--pool_iter_tau_mode', type=str, default='shared', choices=['shared', 'per_stage'])
        parser.add_argument('--pool_iter_tau_init', type=float, default=1.0)
        parser.add_argument('--pool_iter_tau_clamp', type=float, nargs=2, default=(0.1, 10.0))
        parser.add_argument('--ms_heads', type=int, default=8)
        parser.add_argument('--decoder_heads', type=int, default=8)
        parser.add_argument('--decoder_ffn_dim', type=int, default=1024)
        parser.add_argument('--decoder_operation', type=str, default='//', choices=['//', '%'])
        parser.add_argument('--decoder_num_blocks', type=int, nargs='+', default=[1, 1, 1])
        parser.add_argument('--febr_blocks', type=int, nargs='+', default=[1, 1, 1])
        parser.add_argument('--logit_scale_init', type=float, default=10.0)
        parser.add_argument('--min_bin_ratio', type=float, default=1e-4)

        parser.add_argument('--use_task_exchange', action='store_true')
        parser.add_argument('--exchange_levels', type=str, default='5,4,3,2,1')
        parser.add_argument('--exchange_on_fp_boundary', action='store_true')
        parser.add_argument('--exchange_mode', type=str, default='bidirectional', choices=['bidirectional', 'fp2prob', 'prob2fp'])
        parser.add_argument('--exchange_conf_mode', type=str, default='none', choices=['none', 'spatial'])
        parser.add_argument('--exchange_alpha_init', type=float, default=0.01)
        parser.add_argument('--exchange_alpha_prob_init', type=float, default=0.01)
        parser.add_argument('--exchange_alpha_fp_init', type=float, default=0.01)
        parser.add_argument('--exchange_conf_bias_init', type=float, default=3.0)
        parser.add_argument('--fp_zone_kernel_size', type=int, default=3)
        parser.add_argument('--fp_zone_thresh', type=float, default=0.5)

        parser.add_argument('--num_height_bins', type=int, default=64)
        parser.add_argument('--h_min', type=float, default=0.0)
        parser.add_argument('--h_max', type=float, default=145.0)

        parser.add_argument('--ckpt_name', type=str, default='model_best_val_loss')
        parser.add_argument('--use_amp', action='store_true')
        parser.add_argument('--fp_thresh', type=float, default=2.0)
        parser.add_argument('--fp_temp', type=float, default=1.0)

        parser.add_argument('--ablate_no_fp', action='store_true', help='remove footprint branch')
        parser.add_argument('--ablate_no_bins', action='store_true', help='remove FEBR bin head and use direct height head')
        parser.add_argument('--prob_height_head_act', type=str, default='linear', choices=['linear', 'sigmoid_range'])

        parser.add_argument('--data_dir', type=str, default=r'E:/HeightMap/Samples/256_World')
        parser.add_argument('--n_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=8)

        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        opt, _ = parser.parse_known_args()
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt, phase):
        message = '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            default = self.parser.get_default(k)
            comment = '' if v == default else f'\t[default: {default}]'
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        file_name = os.path.join(opt.save_dir, phase + '_opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        opt = self.gather_options()
        print(f'Task name: {opt.name}')
        print(f'Using {opt.mode} mode!')

        norm_path = os.path.join(opt.data_dir, 'normalization_stats.npz')
        opt.norm_path = norm_path if os.path.exists(norm_path) else None

        opt.use_bins = not bool(opt.ablate_no_bins)
        opt.save_dir = os.path.join(opt.result_dir, opt.name)
        os.makedirs(opt.save_dir, exist_ok=True)
        print(f'Results will be logged to: {opt.save_dir}')

        opt.isTrain = self.isTrain
        self.print_options(opt, 'Train' if opt.isTrain else 'Test')

        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            idx = int(str_id)
            if idx >= 0:
                opt.gpu_ids.append(idx)
        if len(opt.gpu_ids) == 0:
            opt.device = torch.device('cpu')
        elif len(opt.gpu_ids) == 1:
            opt.device = torch.device('cuda:0')
        else:
            raise RuntimeError('Multiple GPU devices are not supported in this cleaned version.')
        print(f'Running on device: [{opt.device}]')

        self.opt = opt
        return self.opt
