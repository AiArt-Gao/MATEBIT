import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import util.util as util

from models.networks.base_network import BaseNetwork
from models.networks.architecture import SPADEResnetBlock, MappingNetwork, ResidualBlock, get_nonspade_norm_layer, \
    AdainRestBlocks, Attention
from models.networks.architecture import FeatureGenerator
from models.networks.dynast_transformer import DynamicTransformerBlock
from models.networks.dynast_transformer import DynamicSparseTransformerBlock
from models.networks.modules import SelfAttentionLayer
from models.networks.nceloss import BidirectionalNCE1
from models.networks.position_encoding import PositionEmbeddingSine
from models.networks.ops import dequeue_data, queue_data
from models.networks import calc_contrastive_loss

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class PatchSampleF(nn.Module):
    def __init__(self):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)

    def forward(self, feat, num_patches=64, patch_ids=None):
        # b c h w --> b h w c --> b hw c
        feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
        if patch_ids is not None:
            patch_id = patch_ids
        else:
            patch_id = torch.randperm(feat_reshape.shape[1], device=feat[0].device)
            patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
        x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
        x_sample = self.l2norm(x_sample)
        # return_feats.append(x_sample)
        return x_sample, patch_id


class DynaSTGenerator(BaseNetwork):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(spectual_norm=True)
        parser.add_argument('--max_multi', type=int, default=8)
        parser.add_argument('--top_k', type=int, default=4)
        parser.add_argument('--n_layers', type=int, default=3)
        parser.add_argument('--smooth', type=float, default=0.01)
        parser.add_argument('--pos_dim', type=int, default=16)
        parser.add_argument('--prune_dim', type=int, default=16)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.encoder_q = FeatureGenerator(opt.semantic_nc, opt.ngf, opt.max_multi, norm='instance')
        self.encoder_kv = FeatureGenerator(opt.semantic_nc, opt.ngf, opt.max_multi, norm='instance')
        self.patch_sample = PatchSampleF()
        self.nceloss = BidirectionalNCE1()
        self.feat_size = 64

        # self.pos_layer = PositionEmbeddingSine(opt.pos_dim * min(opt.max_multi, 8) // 2, normalize=True)
        # pos_embed = self.pos_layer(torch.randn(1, opt.pos_dim * min(opt.max_multi, 8), opt.crop_size // 8, opt.crop_size // 8))
        pos_embed = nn.Parameter(torch.randn(
            1, opt.pos_dim * min(opt.max_multi, 8), opt.crop_size // 8, opt.crop_size // 8))
        self.register_parameter('pos_embed', pos_embed)
        self.embed_q4 = SPADEResnetBlock(opt.ngf * min(opt.max_multi, 8), opt.semantic_nc)
        self.embed_kv4 = SPADEResnetBlock(opt.ngf * min(opt.max_multi, 8), opt.semantic_nc)
        if self.opt.isTrain:
            self.queue = torch.zeros((0, self.feat_size), dtype=torch.float).cuda()

        transformer_4_list = []
        for _ in range(opt.n_layers):
            transformer_4_list.append(DynamicTransformerBlock((opt.ngf + opt.pos_dim) * min(opt.max_multi, 8),
                                                              opt.ngf * min(opt.max_multi, 8),
                                                              opt.prune_dim * min(opt.max_multi, 8),
                                                              opt.semantic_nc, smooth=None))
        self.transformer_4 = nn.ModuleList(transformer_4_list)
        self.mapping = MappingNetwork(opt.ngf * min(opt.max_multi, 8), opt.ngf * 4, opt.ngf)
        kw, pw = 3, 1
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer5 = nn.Sequential(
            norm_layer(nn.Conv2d(opt.ngf * 4 * 2, opt.ngf * 4 * 2, kw, stride=2, padding=pw)),
            ResidualBlock(opt.ngf * 4 * 2),
        )
        self.bottleneck = nn.Sequential(
            #  b 512 32 32
            norm_layer(nn.Conv2d(opt.ngf * 4 * 2, opt.ngf * 4 * 2, kw, stride=1, padding=pw)),
            ResidualBlock(opt.ngf * 4 * 2),
        )
        self.final = nn.Conv2d(opt.ngf, 3, 3, stride=1, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.adain1 = AdainRestBlocks(dim=opt.ngf * 4 * 2, dimin=64)
        self.adain2 = AdainRestBlocks(dim=opt.ngf * 4, dimin=64)
        self.adain3 = AdainRestBlocks(dim=opt.ngf * 2, dimin=64)
        self.adain4 = AdainRestBlocks(dim=opt.ngf, dimin=64)

        self.style1 = nn.Sequential(
            norm_layer(nn.Conv2d(opt.ngf * 4 * 4, opt.ngf * 4 * 2, kw, stride=1, padding=pw)),
            ResidualBlock(opt.ngf * 4 * 2),
        )
        self.style2 = nn.Sequential(
            norm_layer(nn.Conv2d(opt.ngf * 4 * 4, opt.ngf * 4, kw, stride=1, padding=pw)),
            ResidualBlock(opt.ngf * 4),
        )
        self.style3 = nn.Sequential(
            norm_layer(nn.Conv2d(opt.ngf * 4 * 2, opt.ngf * 2, kw, stride=1, padding=pw)),
            ResidualBlock(opt.ngf * 2),
        )
        self.attn = Attention(2 * opt.ngf, 'spectral' in opt.norm_G)
        self.style4 = nn.Sequential(
            norm_layer(nn.Conv2d(opt.ngf * 4, opt.ngf, kw, stride=1, padding=pw)),
            ResidualBlock(opt.ngf),
        )
        self.res_block = ResidualBlock(opt.ngf)

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

    def forward(self, ref_img, real_img, seg_map, ref_seg_map):
        out = {}

        out['warp_out'] = []
        adaptive_feature_seg = self.encoder_q(seg_map)
        adaptive_feature_img = self.encoder_kv(ref_img)
        for i in range(len(adaptive_feature_seg)):
            adaptive_feature_seg[i] = util.feature_normalize(adaptive_feature_seg[i])
            adaptive_feature_img[i] = util.feature_normalize(adaptive_feature_img[i])

        if self.opt.isTrain and self.opt.weight_novgg_featpair > 0:
            adaptive_feature_img_pair = self.encoder_kv(real_img)
            loss_novgg_featpair = 0
            weights = [1.0, 1.0, 1.0, 1.0]
            for i in range(len(adaptive_feature_img_pair)):
                adaptive_feature_img_pair[i] = util.feature_normalize(adaptive_feature_img_pair[i])
                loss_novgg_featpair += F.l1_loss(adaptive_feature_seg[i], adaptive_feature_img_pair[i]) * weights[i]
            out['loss_novgg_featpair'] = loss_novgg_featpair * self.opt.weight_novgg_featpair

            # 对应位置 信息 需要改进
            if self.opt.mcl:
                feat_k, sample_ids = self.patch_sample(adaptive_feature_seg[0], 64, None)
                feat_q, _ = self.patch_sample(adaptive_feature_img_pair[0], 64, sample_ids)
                nceloss = self.nceloss(feat_k, feat_q)
                out['nceloss'] = nceloss * self.opt.nce_w

        q4 = self.embed_q4(adaptive_feature_seg[0], seg_map)
        ref_feature = self.embed_kv4(adaptive_feature_img[0], ref_img)

        x4 = q4
        pos = self.pos_embed

        for i in range(self.opt.n_layers):
            x4, warped, cross_cor_map = self.transformer_4[i](
                x4, ref_feature, ref_feature, pos, seg_map, F.avg_pool2d(ref_img, 8, stride=8) if self.opt.isTrain else None)
            if self.opt.isTrain:
                out['warp_out'].append(warped)

        ref_mapping = self.mapping(ref_feature)
        # x warp_feature  unet down up
        x5 = self.layer5(self.actvn(x4))  # b 512 16 16 # 64 128 256
        bottleneck = self.bottleneck(self.actvn(x5))  # b 512 16 16
        up0 = self.style1(torch.cat((bottleneck, x5), dim=1))  # b 1024 -512

        out0 = self.adain1(up0, ref_mapping)
        up1 = self.style2(torch.cat((self.up(out0), x4), dim=1)) # 512  32 32
        out1 = self.adain2(up1, ref_mapping)
        up2 = self.style3(torch.cat((self.up(out1), adaptive_feature_seg[1]), dim=1)) # 256 64 64
        out2 = self.adain3(up2, ref_mapping)

        if self.opt.use_atten:
            out2 = self.attn(out2)

        up3 = self.style4(torch.cat((self.up(out2), adaptive_feature_seg[2]), dim=1)) # 64 128 128
        out3 = self.adain4(up3, ref_mapping)
        x = F.leaky_relu(self.up(out3), 2e-1) #  64 256 256
        x = self.res_block(x)
        x = self.final(x)
        x = torch.tanh(x)

        out['fake_image'] = x

        #　training, 100 epoch start 第 100轮的风格当作 负样例
        if self.opt.isTrain and self.opt.contrastive_weight > 0. and self.opt.epoch > 100:

            if self.opt.epoch <= 101:
                ref_mapping = ref_mapping.detach()
                self.queue = queue_data(self.queue, ref_mapping)
                self.queue = dequeue_data(self.queue, K=1024)
            else:
                ref_mapping = ref_mapping.detach()
                fake_features = self.encoder_kv(x)
                # 调制一下
                fake_feature_x4 = util.feature_normalize(fake_features[0])
                del fake_features
                fake_feature = self.embed_kv4(fake_feature_x4, x)
                fake_mapping = self.mapping(fake_feature) # z
                # ref_mapping z+
                # z- 用个queue存 负样例, 将其他refer 风格当作负样例
                contrastive_loss = calc_contrastive_loss(fake_mapping, ref_mapping, self.queue)
                out['contrastive_loss'] = contrastive_loss * self.opt.contrastive_weight
                self.queue = queue_data(self.queue, ref_mapping)
                self.queue = dequeue_data(self.queue, K=1024)

        return out
