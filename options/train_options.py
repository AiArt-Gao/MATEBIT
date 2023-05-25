# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # for displays
        parser.add_argument('--display_freq', type=int, default=500, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        # for training
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--D_steps_per_G', type=int, default=1, help='number of discriminator iterations per generator iterations.')
        # for discriminators
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')
        parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')
        parser.add_argument('--real_reference_probability', type=float, default=0.7, help='self-supervised training probability')
        parser.add_argument('--hard_reference_probability', type=float, default=0.3, help='hard reference training probability')
        # training loss weights
        parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        parser.add_argument('--weight_contextual', type=float, default=1.0, help='ctx loss weight')
        parser.add_argument('--weight_novgg_featpair', type=float, default=10.0,
                            help='in no vgg setting, use pair feat loss in domain adaptation')
        parser.add_argument('--mcl', action='store_true', help='Use marginal contrastive learning')
        parser.add_argument('--nce_w', type=float, default=0.01, help='weight for marginal contrastive loss')
        parser.add_argument('--style_weight', type=float, default=1.0, help='weight for marginal contrastive loss')
        parser.add_argument('--contrastive_weight', type=float, default=0., help='weight for marginal contrastive loss')


        self.isTrain = True
        return parser
