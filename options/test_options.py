from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--phase_test', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--phase1', type=str, default='test1', help='train, val, test, etc')
        parser.add_argument('--phase2', type=str, default='test2', help='train, val, test, etc')
        parser.add_argument('--phase3', type=str, default='test3', help='train, val, test, etc')
        parser.add_argument('--phase4', type=str, default='test4', help='train, val, test, etc')
        parser.add_argument('--phase5', type=str, default='test5', help='train, val, test, etc')
        parser.add_argument('--phase6', type=str, default='test6', help='train, val, test, etc')
        parser.add_argument('--phase7', type=str, default='test7', help='train, val, test, etc')
        parser.add_argument('--phase8', type=str, default='test8', help='train, val, test, etc')
        parser.add_argument('--phase9', type=str, default='test9', help='train, val, test, etc')
        parser.add_argument('--phase10', type=str, default='test10', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=800, help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
