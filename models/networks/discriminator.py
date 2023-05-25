# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn as nn
import torch.nn.functional as F

from models.networks.base_network import BaseNetwork
import util.util as util


class MultiscaleDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netD_subarch', type=str, default='n_layer',
                            help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2,
                            help='number of discriminators to be used in multiscale')
        opt, _ = parser.parse_known_args()
        # define properties of each discriminator of the multiscale discriminator
        subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator', \
                                            'models.networks.discriminator')
        subnetD.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt):
        subarch = opt.netD_subarch
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(opt)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)
        return result


class NLayerDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4, help='# layers in each discriminator')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        kw = 4
        padw = int((kw - 1.0) / 2)
        nf = opt.ndf
        input_nc = self.compute_D_input_nc(opt)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]
        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            if n == opt.n_layers_D - 1:
                dec = []
                nc_dec = nf_prev
                for _ in range(opt.n_layers_D - 1):
                    dec += [nn.Upsample(scale_factor=2),
                            spectral_norm(nn.Conv2d(nc_dec, int(nc_dec//2), kernel_size=3, stride=1, padding=1)),
                            nn.InstanceNorm2d(int(nc_dec//2)),
                            nn.LeakyReLU(0.2, False)]
                    nc_dec = int(nc_dec // 2)
                dec += [nn.Conv2d(nc_dec, opt.semantic_nc, kernel_size=3, stride=1, padding=1)]
                self.dec = nn.Sequential(*dec)
            sequence += [[spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw)),
                          nn.InstanceNorm2d(nf),
                          nn.LeakyReLU(0.2, False)]]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self, opt):
        input_nc = opt.label_nc + opt.output_nc
        # if opt.contain_dontcare_label:
        #     input_nc += 1
        return input_nc

    def forward(self, input):
        results = [input]
        seg = None
        cam_logit = None
        for name, submodel in self.named_children():
            if 'model' not in name:
                continue
            x = results[-1]
            intermediate_output = submodel(x)
            results.append(intermediate_output)
        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            retu = results[1:]
        else:
            retu = results[-1]
        return retu
