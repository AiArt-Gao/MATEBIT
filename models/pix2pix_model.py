# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
import models.networks as networks
import util.util as util
import itertools

try:
    from torch.cuda.amp import autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        if is_train:
            parser.add_argument('--weight_warp_self', type=float, default=1000.0, help='push warp self to ref')
            parser.add_argument('--weight_gan', type=float, default=10.0, help='weight of all loss in stage1')
            parser.add_argument('--no_ganFeat_loss', action='store_true',
                                help='if specified, do *not* use discriminator feature matching loss')
            parser.add_argument('--weight_ganFeat', type=float, default=10.0, help='weight for feature matching loss')
            parser.add_argument('--which_perceptual', type=str, default='4_2', help='relu5_2 or relu4_2')
            parser.add_argument('--weight_perceptual', type=float, default=0.001)
            parser.add_argument('--weight_vgg', type=float, default=10.0, help='weight for vgg loss')
            parser.add_argument('--weight_fm_ratio', type=float, default=1.0)
            parser.add_argument('--vgg_path', type=str, required=True)
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor
        self.net = torch.nn.ModuleDict(self.initialize_networks(opt))
        device = torch.device('cuda:%d' % self.opt.gpu_ids[0] if self.use_gpu() else 'cpu')
        # set loss functions
        if opt.isTrain:
            # vgg network
            self.vggnet_fix = networks.architecture.VGG19_feature_color_torchversion(
                vgg_normal_correct=opt.vgg_normal_correct)
            self.vggnet_fix.load_state_dict(torch.load('vgg/vgg19_conv.pth'))
            self.vggnet_fix.eval()
            for param in self.vggnet_fix.parameters():
                param.requires_grad = False
            self.vggnet_fix.to(device)
            # GAN loss
            self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            # contextual loss
            self.contextual_forward_loss = networks.ContextualLoss_forward(opt)
            # L1 loss
            self.criterionFeat = torch.nn.L1Loss()
            # L2 loss
            self.MSE_loss = torch.nn.MSELoss()
            # setting which layer is used in the perceptual loss
            # style loss gram
            self.StyleLoss = networks.StyleLoss()

            if opt.which_perceptual == '5_2':
                self.perceptual_layer = -1
            elif opt.which_perceptual == '4_2':
                self.perceptual_layer = -2

    def forward(self, data, mode, GforD=None):
        input_label, input_semantics, real_image, self_ref, ref_image, ref_label, ref_semantics = self.preprocess_input(
            data.copy())

        if mode == 'generator':
            g_loss, generated_out = self.compute_generator_loss(input_label,
                                                                input_semantics, real_image, ref_label,
                                                                ref_semantics, ref_image, self_ref)
            out = {'fake_image': generated_out['fake_image'], 'input_semantics': input_semantics,
                   'ref_semantics': ref_semantics,
                   'warp_out': None if 'warp_out' not in generated_out else generated_out['warp_out'],
                   'loss_novgg_featpair': None if 'loss_novgg_featpair' not in generated_out else generated_out['loss_novgg_featpair'],
                   'nceloss': None if 'nceloss' not in generated_out else generated_out['nceloss'],
                   'contrastive_loss': None if 'contrastive_loss' not in generated_out else generated_out['contrastive_loss'],
                   }
            return g_loss, out

        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(input_semantics,
                                                     real_image, GforD, label=input_label)
            return d_loss

        elif mode == 'inference':
            with torch.no_grad():
                out = self.inference(input_semantics, ref_semantics=ref_semantics,
                                     ref_image=ref_image, self_ref=self_ref,
                                     real_image=real_image)
            out['input_semantics'] = input_semantics
            out['ref_semantics'] = ref_semantics
            return out

        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2
        optimizer_G = torch.optim.Adam(itertools.chain(self.net['netG'].parameters()), lr=G_lr, betas=(beta1, beta2),
                                       eps=1e-3)
        optimizer_D = torch.optim.Adam(itertools.chain(self.net['netD'].parameters()),
                                       lr=D_lr, betas=(beta1, beta2))
        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.net['netG'], 'G', epoch, self.opt)
        util.save_network(self.net['netD'], 'D', epoch, self.opt)

    def initialize_networks(self, opt):
        net = {'netG': networks.define_G(opt), 'netD': networks.define_D(opt) if opt.isTrain else None}
        if not opt.isTrain or opt.continue_train:
            net['netG'] = util.load_network(net['netG'], 'G', opt.which_epoch, opt)
            if opt.isTrain:
                net['netD'] = util.load_network(net['netD'], 'D', opt.which_epoch, opt)
        return net

    def preprocess_input(self, data):
        if self.opt.dataset_mode == 'celebahq':
            glasses = data['label'][:, 1::2, :, :].long()
            data['label'] = data['label'][:, ::2, :, :]
            glasses_ref = data['label_ref'][:, 1::2, :, :].long()
            data['label_ref'] = data['label_ref'][:, ::2, :, :]
            if self.use_gpu():
                glasses = glasses.cuda()
                glasses_ref = glasses_ref.cuda()
        else:
            if self.opt.dataset_mode == 'celebahqedge':
                input_semantics = data['label'].clone().float()
                data['label'] = data['label'][:, :1, :, :]
                ref_semantics = data['label_ref'].clone().float()
                data['label_ref'] = data['label_ref'][:, :1, :, :]
            elif self.opt.dataset_mode == 'deepfashion':
                input_semantics = data['label'].clone().float()
                data['label'] = data['label'][:, :3, :, :]
                ref_semantics = data['label_ref'].clone().float()
                data['label_ref'] = data['label_ref'][:, :3, :, :]

            elif self.opt.dataset_mode == 'ade20klayout':
                input_semantics = data['label'][:, 3:, :, :].clone().float()
                data['label'] = data['label'][:, :3, :, :]
                ref_semantics = data['label_ref'][:, 3:, :, :].clone().float()
                data['label_ref'] = data['label_ref'][:, :3, :, :]

            elif self.opt.dataset_mode == 'cocolayout':
                input_semantics = data['label'][:, 3:, :, :].clone().float()
                data['label'] = data['label'][:, :3, :, :]
                ref_semantics = data['label_ref'][:, 3:, :, :].clone().float()
                data['label_ref'] = data['label_ref'][:, :3, :, :]
            else:# matfaces
                input_semantics = data['label'][:, :3, :, :].clone().float()
                data['label'] = data['label'][:, :3, :, :]
                ref_semantics = data['label_ref'][:, :3, :, :].clone().float()
                data['label_ref'] = data['label_ref'][:, :3, :, :]
        # move to GPU and change data types
        if self.opt.dataset_mode != 'deepfashion':
            data['label'] = data['label'].long()
            data['label_ref'] = data['label_ref'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['image'] = data['image'].cuda()
            data['ref'] = data['ref'].cuda()
            data['label_ref'] = data['label_ref'].cuda()
            data['self_ref'] = data['self_ref'].cuda()

        # create one-hot label map
        if self.opt.dataset_mode == 'ade20k' or self.opt.dataset_mode == 'coco':
            label_map = data['label']
            bs, _, h, w = label_map.size()
            nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
                else self.opt.label_nc
            input_label = self.FloatTensor(bs, nc, h, w).zero_()
            input_semantics = input_label.scatter_(1, label_map, 1.0)

            label_map = data['label_ref']
            label_ref = self.FloatTensor(bs, nc, h, w).zero_()
            ref_semantics = label_ref.scatter_(1, label_map, 1.0)

        if self.use_gpu():
            input_semantics = input_semantics.cuda()
            ref_semantics = ref_semantics.cuda()

        if self.opt.dataset_mode == 'celebahq':
            assert input_semantics[:, -3:-2, :, :].sum().cpu().item() == 0
            input_semantics[:, -3:-2, :, :] = glasses
            assert ref_semantics[:, -3:-2, :, :].sum().cpu().item() == 0
            ref_semantics[:, -3:-2, :, :] = glasses_ref

        return data['label'], input_semantics, data['image'], data['self_ref'], data['ref'], data[
            'label_ref'], ref_semantics

    def get_ctx_loss(self, source, target):
        contextual_style5_1 = torch.mean(self.contextual_forward_loss(source[-1], target[-1].detach())) * 8
        contextual_style4_1 = torch.mean(self.contextual_forward_loss(source[-2], target[-2].detach())) * 4
        contextual_style3_1 = torch.mean(self.contextual_forward_loss(F.avg_pool2d(source[-3], 2), F.avg_pool2d(target[-3].detach(), 2))) * 2
        return contextual_style5_1 + contextual_style4_1 + contextual_style3_1

    def get_style_loss(self, source, target):
        contextual_style = self.StyleLoss(source, target)
        return contextual_style

    def compute_generator_loss(self, input_label, input_semantics, real_image, ref_label=None, ref_semantics=None,
                               ref_image=None, self_ref=None):
        G_losses = {}
        generate_out = self.generate_fake(input_semantics, real_image, ref_semantics=ref_semantics, ref_image=ref_image,
                                          self_ref=self_ref)
        generate_out['fake_image'] = generate_out['fake_image'].float()
        weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        sample_weights = self_ref / (sum(self_ref) + 1e-5)
        sample_weights = sample_weights.view(-1, 1, 1, 1)

        if 'loss_novgg_featpair' in generate_out and generate_out['loss_novgg_featpair'] is not None:
            G_losses['no_vgg_feat'] = generate_out['loss_novgg_featpair']
        """warping loss"""
        if self.opt.weight_warp_self > 0:
            """512x512"""
            warps = generate_out['warp_out']
            G_losses['G_warp_self'] = 0.
            for warp in warps:
                scale_factor = real_image.size()[-1] // warp.size()[-1]
                gt = F.avg_pool2d(real_image, scale_factor, stride=scale_factor)
                G_losses['G_warp_self'] += torch.mean(F.l1_loss(
                    warp, gt, reduction='none') * sample_weights) * self.opt.weight_warp_self
        """gan loss"""
        pred_fake, pred_real = self.discriminate(input_semantics, generate_out['fake_image'], real_image)
        G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False) * self.opt.weight_gan
        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = 0.0
            for i in range(num_D):
                # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):
                    # for each layer output
                    unweighted_loss = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.weight_ganFeat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss
        """feature matching loss"""
        fake_features = self.vggnet_fix(generate_out['fake_image'], ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)
        loss = 0
        for i in range(len(generate_out['real_features'])):
            loss += weights[i] * util.weighted_l1_loss(fake_features[i], generate_out['real_features'][i].detach(),
                                                       sample_weights)
        G_losses['fm'] = loss * self.opt.weight_vgg * self.opt.weight_fm_ratio
        """perceptual loss"""
        feat_loss = util.mse_loss(fake_features[self.perceptual_layer],
                                  generate_out['real_features'][self.perceptual_layer].detach())
        G_losses['contextual'] = self.get_ctx_loss(fake_features, generate_out[
            'ref_features']) * self.opt.weight_vgg * self.opt.weight_contextual
        G_losses['perc'] = feat_loss * self.opt.weight_perceptual

        if self.opt.style_weight > 0:
            G_losses['style_loss'] = self.get_style_loss(fake_features,
                                                         generate_out['ref_features']) * self.opt.style_weight * self.opt.weight_vgg
        if self.opt.contrastive_weight > 0 and self.opt.epoch > 101:
            G_losses['contrastive_loss'] = generate_out['contrastive_loss']

        if self.opt.mcl:
            G_losses['nceloss'] = generate_out['nceloss']

        return G_losses, generate_out

    def compute_discriminator_loss(self, input_semantics, real_image, GforD, label=None):
        D_losses = {}
        with torch.no_grad():
            fake_image = GforD['fake_image'].detach()
            fake_image.requires_grad_()
        pred_fake, pred_real = self.discriminate(input_semantics, fake_image, real_image)
        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False, for_discriminator=True) * self.opt.weight_gan
        D_losses['D_real'] = self.criterionGAN(pred_real, True, for_discriminator=True) * self.opt.weight_gan
        return D_losses

    def generate_fake(self, input_semantics, real_image, ref_semantics=None, ref_image=None, self_ref=None):
        generate_out = {}
        generate_out['ref_features'] = self.vggnet_fix(ref_image, ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)
        generate_out['real_features'] = self.vggnet_fix(real_image, ['r12', 'r22', 'r32', 'r42', 'r52'],
                                                        preprocess=True)
        with autocast(enabled=self.opt.amp):
            network_out = self.net['netG'](ref_image, real_image, input_semantics, ref_semantics)
        generate_out = {**generate_out, **network_out}
        return generate_out

    def inference(self, input_semantics, ref_semantics=None, ref_image=None, self_ref=None, real_image=None):
        with autocast(enabled=self.opt.amp):
            network_out = self.net['netG'](ref_image, real_image, input_semantics, ref_semantics)
        return network_out

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        with autocast(enabled=self.opt.amp):
            discriminator_out = self.net['netD'](fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)
        return pred_fake, pred_real

    def divide_pred(self, pred):
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]
        return fake, real

    def use_gpu(self):
        return self.opt.gpu_ids[0] >= 0
