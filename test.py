# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torchvision.utils import save_image
import os
import data
from util.util import mkdir
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel


"""
# test

python test.py  --PONO --PONO_C --vgg_normal_correct  --video_like  --nThreads 16 --display_winsize 256 --load_size 256  --crop_size 256  --label_nc 3 --batchSize 4  --gpu_ids 0 --netG dynast --use_atten  --n_layers 3 

"""
if __name__ == '__main__':
    opt = TestOptions().parse()
    dataloader = data.create_dataloader(opt)
    model = Pix2PixModel(opt)
    if len(opt.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    else:
        model.to(opt.gpu_ids[0])
    model.eval()
    save_root = os.path.join(opt.checkpoints_dir, opt.name, 'test')
    mkdir(save_root)
    if opt.save_per_img:
        mkdir(os.path.join(save_root, 'fake'))
        mkdir(os.path.join(save_root, 'real'))
        mkdir(os.path.join(save_root, 'label'))
        mkdir(os.path.join(save_root, 'ref'))
    for i, data_i in enumerate(dataloader):
        print('{} / {}'.format(i, len(dataloader)))
        if i * opt.batchSize >= opt.how_many:
            break
        imgs_num = data_i['label'].shape[0]
        out = model(data_i, mode='inference')

        if opt.dataset_mode == 'celebahq':
            data_i['label'] = data_i['label'][:, ::2, :, :]
            data_i['label_ref'] = data_i['label_ref'][:, ::2, :, :]
        elif opt.dataset_mode == 'celebahqedge':
            data_i['label'] = data_i['label'][:, :1, :, :]
            data_i['label_ref'] = data_i['label_ref'][:, :1, :, :]
        elif opt.dataset_mode == 'deepfashion':
            data_i['label'] = data_i['label'][:, :3, :, :]
            data_i['label_ref'] = data_i['label_ref'][:, :3, :, :]
        if data_i['label'].shape[1] == 3:
            label = data_i['label']
            label_ref = data_i['label_ref']
        else:
            label = data_i['label'].expand(-1, 3, -1, -1).float() / data_i['label'].max()
            label_ref = data_i['label_ref'].expand(-1, 3, -1, -1).float() / data_i['label_ref'].max()
        if opt.save_per_img:
            try:
                for it in range(imgs_num):
                    if opt.dataset_mode == 'deepfashion':
                        _, name = data_i['path'][it].split(
                            opt.dataroot + '/' if not opt.dataroot.endswith('/') else opt.dataroot)
                        name = name.replace('/', '_')
                    else:
                        name = os.path.basename(data_i['path'][it])
                    save_name = os.path.join(save_root, 'fake', name)
                    save_image((out['fake_image'][it:it + 1] + 1) / 2, save_name, padding=0, normalize=False)
                    save_name = os.path.join(save_root, 'real', name)
                    save_image((data_i['image'][it:it + 1] + 1) / 2, save_name, padding=0, normalize=False)
                    save_name = os.path.join(save_root, 'ref', name)
                    save_image((data_i['ref'][it:it + 1] + 1) / 2, save_name, padding=0, normalize=False)
                    save_name = os.path.join(save_root, 'label', name)
                    save_image(label[it:it + 1], save_name, padding=0, normalize=False)
            except OSError as err:
                print(err)
        else:
            imgs = torch.cat((label.cpu(), data_i['ref'].cpu(), out['fake_image'].data.cpu(), data_i['image'].cpu()), 0)
            try:
                save_name = os.path.join(save_root, '%08d.jpg' % i)
                save_image(imgs, save_name, nrow=imgs_num, padding=0, normalize=True)
            except OSError as err:
                print(err)
