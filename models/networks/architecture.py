# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from util.util import vgg_preprocess
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d
from timm.models.layers import DropPath

def get_nonspade_norm_layer(opt, norm_type='instance'):
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]
        else:
            subnorm_type =norm_type
        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)
        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)
        return nn.Sequential(layer, norm_layer)
    return add_norm_layer
class Attention(nn.Module):
    def __init__(self, ch, use_sn):
        super(Attention, self).__init__()
        # Channel multiplier
        self.ch = ch
        self.theta = nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = nn.Conv2d(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = nn.Conv2d(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        if use_sn:
            self.theta = spectral_norm(self.theta)
            self.phi = spectral_norm(self.phi)
            self.g = spectral_norm(self.g)
            self.o = spectral_norm(self.o)
        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y=None):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2,2])
        g = F.max_pool2d(self.g(x), [2,2])
        # Perform reshapes
        theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x

class AdaIN(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(AdaIN, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, y_mean, y_var):
        # x: N x C x W x H
        size = x.size()
        assert (len(size) == 4)
        b, c = x.shape[:2]
        feat_x = x.view(b, c, -1)
        x_mean = feat_x.mean(dim=2).view(b, c, 1, 1)
        varx = torch.clamp((feat_x * feat_x).mean(dim=2).view(b, c, 1, 1) - x_mean * x_mean, min=0)
        varx = torch.rsqrt(varx + self.epsilon)
        x = (x - x_mean) / varx

        return x * y_var + y_mean

class AdainRestBlocks(nn.Module):
    def __init__(self, dim, dimin, use_bias=False):
        super(AdainRestBlocks, self).__init__()
        self.epsilon = 1e-8
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = AdaIN()
        self.relu1 = nn.ReLU(inplace=True)
        # self.relu1 = nn.ReLU(inplace=True) # v1 mapping  +styleModulator
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = AdaIN()
        self.styleModulator = nn.Linear(dimin, 2*dim)
        # self.mlp_gamma = nn.Linear(dimin, dim)
        # self.mlp_beta = nn.Linear(dimin, dim)
        self.dim = dim
        with torch.no_grad():
            # self.mlp_gamma.weight *= 0.25
            # self.mlp_gamma.bias.data.fill_(0)
            # self.mlp_beta.weight *= 0.25
            # self.mlp_beta.bias.data.fill_(0)
            self.styleModulator.weight *= 0.25
            self.styleModulator.bias.data.fill_(0)

    def forward(self, x, y):
        # Adapt style
        batchSize, nChannel, width, height = x.size()
        styleY = self.styleModulator(y)
        y_var = styleY[:, :self.dim].view(batchSize, self.dim, 1, 1)
        y_mean = styleY[:, self.dim:].view(batchSize, self.dim, 1, 1)
        # y_gamma = self.mlp_gamma(y).view(batchSize, self.dim, 1, 1)
        # y_beta = self.mlp_beta(y).view(batchSize, self.dim, 1, 1)
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, y_mean, y_var)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, y_mean, y_var)
        return out + x

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h

        return out

def PositionalNorm2d(x, epsilon=1e-8):
    # x: B*C*W*H normalize in C dim
    mean = x.mean(dim=1, keepdim=True)
    std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
    output = (x - mean) / std
    return output


class SPADE(nn.Module):

    def __init__(self, dim, ic, ks=3):
        super().__init__()
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(ic, dim, kernel_size=(ks, ks), padding=ks // 2, padding_mode='reflect'),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(dim, dim, kernel_size=(ks, ks), padding=ks // 2, padding_mode='reflect')
        self.mlp_beta = nn.Conv2d(dim, dim, kernel_size=(ks, ks), padding=ks // 2, padding_mode='reflect')
        self.coord = CoordAtt(dim, dim)
    def forward(self, x, signal):

        if signal.shape[-2:] != x.shape[-2:]:
            signal = F.interpolate(signal, x.shape[-2:])
        hidden = self.mlp_shared(signal)
        gamma = self.mlp_gamma(hidden)
        beta = self.mlp_beta(hidden)
        # pono 用 coordatten 替换
        return (1 + gamma) * PositionalNorm2d(x) + beta


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class AdaBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.fast_path = nn.Sequential(
            LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, dim)
        )
        self.fast_path_gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                            requires_grad=True) if layer_scale_init_value > 0 else None

    def forward_ffn(self, x):
        #b c h w -- > b h w c
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        return x

    def forward(self, x, mask=None):
        input_x = x
        if mask is None:  # compatible with the original implementation
            x = self.dwconv(x)
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
            x = self.forward_ffn(x)
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            x = input_x + self.drop_path(x)
            return x

class ResidualBlock(nn.Module):
    def __init__(self, dim, ks=3):
        super(ResidualBlock, self).__init__()
        self.relu = nn.PReLU()
        self.model = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(ks, ks), padding=ks // 2, stride=(1, 1), padding_mode='reflect'),
            nn.InstanceNorm2d(dim),
            self.relu,
            nn.Conv2d(dim, dim, kernel_size=(ks, ks), padding=ks // 2, stride=(1, 1), padding_mode='reflect'),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        out = self.relu(x + self.model(x))
        return out


class SPADEResnetBlock(nn.Module):
    def __init__(self, dim, ic, use_spectral_norm=True, ks=3):
        super().__init__()
        # Attributes

        self.conv_0 = nn.Conv2d(dim, dim, kernel_size=(ks, ks), padding=ks // 2, padding_mode='reflect')
        self.conv_1 = nn.Conv2d(dim, dim, kernel_size=(ks, ks), padding=ks // 2, padding_mode='reflect')

        if use_spectral_norm:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)

        self.norm_0 = SPADE(dim, ic)
        self.norm_1 = SPADE(dim, ic)
        self.relu = nn.PReLU()

    def forward(self, x, seg):
        dx = self.conv_0(self.relu(self.norm_0(x, seg)))
        dx = self.conv_1(self.relu(self.norm_1(dx, seg)))
        out = self.relu(x + dx)
        return out

class MappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super(MappingNetwork, self).__init__()
        self.network = nn.Sequential(nn.Linear(z_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(map_hidden_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(map_hidden_dim, map_output_dim))

        self.network.apply(self.kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, features):

        z = self.avgpool(features).view(features.shape[0], -1)
        mapping_codes = self.network(z)
        return mapping_codes

    def kaiming_leaky_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


class FeatureGenerator(nn.Module):

    def __init__(self, ic, nf=64, max_multi=4, kw=3, use_spectral_norm=True, norm='batch'):
        # Can be changed to patch projection in the future
        super().__init__()
        conv_1 = nn.Conv2d(ic, nf, (3, 3), (1, 1), padding=1, padding_mode='reflect')
        self.layer1 = nn.Sequential(
            spectral_norm(conv_1) if use_spectral_norm else conv_1,
            nn.InstanceNorm2d(nf),
            ResidualBlock(nf, ks=kw),
        )
        conv_2 = nn.Conv2d(nf, nf * min(2, max_multi), (3, 3),
                           stride=(2, 2), padding=1, padding_mode='reflect')
        self.layer2 = nn.Sequential(
            spectral_norm(conv_2) if use_spectral_norm else conv_2,
            nn.InstanceNorm2d(nf * min(2, max_multi)),
            ResidualBlock(nf * min(2, max_multi), ks=kw),
        )
        conv_3 = nn.Conv2d(nf * min(2, max_multi), nf * min(4, max_multi), (3, 3),
                           stride=(2, 2), padding=1, padding_mode='reflect')
        self.layer3 = nn.Sequential(
            spectral_norm(conv_3) if use_spectral_norm else conv_3,
            nn.InstanceNorm2d(nf * min(4, max_multi)),
            ResidualBlock(nf * min(4, max_multi), ks=kw),
        )
        conv_4 = nn.Conv2d(nf * min(4, max_multi), nf * min(8, max_multi), (3, 3),
                           stride=(2, 2), padding=1, padding_mode='reflect')
        self.layer4 = nn.Sequential(
            spectral_norm(conv_4) if use_spectral_norm else conv_4,
            nn.InstanceNorm2d(nf * min(8, max_multi)),
            ResidualBlock(nf * min(8, max_multi), ks=kw),
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x4, x3, x2, x1]


class EmbeddingLayer(nn.Module):

    def __init__(self, ic, patch_size, dim, prev_dim=0, active=nn.LeakyReLU()):
        super().__init__()
        self.conv = nn.Conv2d(patch_size * patch_size * ic + ic + prev_dim, dim, (1, 1))
        self.patch_size = patch_size
        self.active = active

    def forward(self, x, prev_layer=None):
        b, c, h, w = x.shape
        x_patch = x.view(b, c, h // self.patch_size, self.patch_size, w // self.patch_size, self.patch_size)
        x_patch = x_patch.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, -1, h // self.patch_size, w // self.patch_size)
        x_down = F.avg_pool2d(x, self.patch_size, stride=self.patch_size)
        data = [x_patch, x_down]
        if prev_layer is not None:
            data.append(F.interpolate(prev_layer, (h // self.patch_size, w // self.patch_size), mode='bilinear'))
        return self.active(self.conv(torch.cat(data, dim=1)))


class EmbeddingInverseLayer(nn.Module):

    def __init__(self, patch_size, dim, oc=3, active=nn.Tanh()):
        super().__init__()
        self.conv = nn.Conv2d(dim, patch_size * patch_size * oc, (1, 1))
        self.patch_size = patch_size
        self.active = active

    def forward(self, x):
        b, c, h, w = x.shape
        return self.active(self.conv(x).view(b, self.patch_size, self.patch_size, -1, h, w).permute(
            0, 3, 4, 1, 5, 2).contiguous().view(b, -1, h * self.patch_size, w * self.patch_size))


class VGG19_feature_color_torchversion(nn.Module):
    """
    NOTE: there is no need to pre-process the input 
    input tensor should range in [0,1]
    """
    def __init__(self, pool='max', vgg_normal_correct=False, ic=3):
        super(VGG19_feature_color_torchversion, self).__init__()
        self.vgg_normal_correct = vgg_normal_correct

        self.conv1_1 = nn.Conv2d(ic, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys, preprocess=True):
        ''' 
        NOTE: input tensor should range in [0,1]
        '''
        out = {}
        if preprocess:
            x = vgg_preprocess(x, vgg_normal_correct=self.vgg_normal_correct)
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]
