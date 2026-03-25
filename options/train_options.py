from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--n_epochs', type=int, default=100)
        parser.add_argument('--lr', type=float, default=2e-4)
        parser.add_argument('--lr_policy', type=str, default='step', choices=['step', 'plateau', 'cosine', 'warmcos'])
        parser.add_argument('--lr_decay_freq', type=int, default=5)
        parser.add_argument('--lr_gamma', type=float, default=0.5)
        parser.add_argument('--beta1', type=float, default=0.5)
        parser.add_argument('--early_stop', action='store_true')
        parser.add_argument('--es_patience', type=int, default=5)
        parser.add_argument('--grad_clip', type=float, default=5.0)

        parser.add_argument('--loss', type=str, default='l1+bce')
        parser.add_argument('--lambda_seg', type=float, default=0.0)

        parser.add_argument('--use_fp_weight', action='store_true')
        parser.add_argument('--inner_w', type=float, default=1.0)
        parser.add_argument('--boundary_w', type=float, default=0.5)
        parser.add_argument('--background_w', type=float, default=0.3)
        parser.add_argument('--use_fp_weight_seg', action='store_true')
        parser.add_argument('--inner_w_seg', type=float, default=1.0)
        parser.add_argument('--boundary_w_seg', type=float, default=2.0)
        parser.add_argument('--background_w_seg', type=float, default=0.5)

        parser.add_argument('--max_train_size', type=int, default=None)
        parser.add_argument('--max_val_size', type=int, default=None)
        parser.add_argument('--save_freq', type=int, default=20)
        parser.add_argument('--resume', action='store_true')

        self.isTrain = True
        return parser
