from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--max_test_size', type=int, default=None)
        parser.add_argument('--separate', action='store_true')
        parser.add_argument('--save_fig', action='store_true')
        parser.add_argument('--save_head', '--save_heads', action='store_true', dest='save_head')
        parser.add_argument('--save_mid_feats', action='store_true')
        self.isTrain = False
        return parser
