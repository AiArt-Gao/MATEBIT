import os
import sys
import numpy as np
from PIL import Image
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

def vgg_preprocess(tensor, vgg_normal_correct=False):
    if vgg_normal_correct:
        tensor = (tensor + 1) / 2
    # input is RGB tensor which ranges in [0,1]
    # output is BGR tensor which ranges in [0,255]
    tensor_bgr = torch.cat((tensor[:, 2:3, :, :], tensor[:, 1:2, :, :], tensor[:, 0:1, :, :]), dim=1)
    # tensor_bgr = tensor[:, [2, 1, 0], ...]
    tensor_bgr_ml = tensor_bgr - torch.Tensor([0.40760392, 0.45795686, 0.48501961]).type_as(tensor_bgr).view(1, 3, 1, 1)
    tensor_rst = tensor_bgr_ml * 255
    return tensor_rst

class VGG19_feature_color_torchversion(nn.Module):
    ''' 
    NOTE: there is no need to pre-process the input 
    input tensor should range in [0,1]
    '''

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

class Dataset_(torch.utils.data.Dataset):
    def __init__(self, folder, dataset_mode):
        self.folder = folder
        self.dataset_mode = dataset_mode
        if dataset_mode == 'ade20k':
            self.label_folder = '/mnt/blob/Dataset/ADEChallengeData2016/images/validation'
            self.ref_folder = '/mnt/blob/Dataset/ADEChallengeData2016/images/training'
            self.ref_label_folder = '/mnt/blob/Dataset/ADEChallengeData2016/images/training'
            ref_pair_txt = '/mnt/blob/Code/SPADE_Exemplar/data/ade20k_ref_test_from_train.txt'
        elif 'celebahq' in dataset_mode:
            self.label_folder = '/data/jiangchang/CelebAMask-HQ/all_parts_except_glasses'
            self.ref_folder = '/data/jiangchang/CelebAMask-HQ/CelebA-HQ-img'
            self.ref_label_folder = '/data/jiangchang/CelebAMask-HQ/all_parts_except_glasses'
            ref_pair_txt = '/data/jiangchang/CelebAMask-HQ/celebahq_ref_test.txt'
        elif dataset_mode == 'flickr':
            self.label_folder = '/mnt/blob/Dataset/Flickr/test/mask'
            self.ref_folder = '/mnt/blob/Dataset/Flickr/images'
            self.ref_label_folder = '/mnt/blob/Dataset/Flickr/mask'
            ref_pair_txt = '/mnt/blob/Code/SPADE_Exemplar/data/flickr_ref_test_from_train.txt'
        elif dataset_mode == 'deepfashion':
            self.label_folder = '/mnt/blob/Dataset/DeepFashion/parsing'
            self.ref_folder = '/mnt/blob/Dataset/DeepFashion'
            self.ref_label_folder = '/mnt/blob/Dataset/DeepFashion/test_ref_parsing'
            ref_pair_txt = '/mnt/blob/Code/SPADE_Exemplar/data/deepfashion_ref_test_from_train.txt'
        elif dataset_mode == "metfaces":
            self.label_folder = '/data/jiangchang/Metfaces/edge'
            self.ref_folder = '/data/jiangchang/Metfaces/metfaces/metface'
            self.ref_label_folder = '/data/jiangchang/Metfaces/edge'
            ref_pair_txt = '/data/jiangchang/Metfaces/cocosnet/metface_ref_test.txt'
        elif dataset_mode == "aahq":
            self.label_folder = '/data/jiangchang/aahq/aahq_edge'
            self.ref_folder = '/data/jiangchang/aahq/aahq_data'
            self.ref_label_folder = '/data/jiangchang/aahq/aahq_edge'
            ref_pair_txt = '/data/jiangchang/aahq/dict/aahq_ref_test.txt'

        with open(ref_pair_txt) as fd:
            lines = fd.readlines()
        label_paths = []
        img_paths = []
        ref_label_paths = []
        ref_img_paths = []
        for i in range(len(lines)):
            items = lines[i].strip().split(',')
            if dataset_mode == 'deepfashion':
                items[0] = items[0].replace('\\', '_')
            if not os.path.exists(os.path.join(self.folder, items[0])):
                items[0] = items[0].replace('.jpg', '.png')
            if not os.path.exists(os.path.join(self.folder, items[0])):
                print(items[0] + ' not find!')
            else:
                label_paths.append(os.path.join(self.label_folder, self.tolabel_path(items[0])))
                img_paths.append(os.path.join(self.folder, items[0]))
                ref_label_paths.append(os.path.join(self.ref_label_folder, self.tolabel_path(items[2])))
                ref_img_paths.append(os.path.join(self.ref_folder, items[2]))
        self.label_paths = label_paths
        self.img_paths = img_paths
        self.ref_label_paths = ref_label_paths
        self.ref_img_paths = ref_img_paths
        
        self.transform_img = transforms.Compose([transforms.Resize((256, 256)),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5),
                                                                    (0.5, 0.5, 0.5))])
        self.transform_label = transforms.ToTensor()

    def tolabel_path(self, path):
        if 'celebahq' in self.dataset_mode:
            name, ext = path.split('.')
            path = name.zfill(5) + '.' + ext
        if 'deepfashion' in self.dataset_mode:
            path = path.replace('\\', '_')
        return path.replace('.jpg', '.png')

    def __getitem__(self, index):
        label = Image.open(self.label_paths[index]).convert('L').resize((256, 256), Image.NEAREST)
        label = self.transform_label(label)*255
        img = Image.open(self.img_paths[index]).convert('RGB')
        img = self.transform_img(img)
        ref_label = Image.open(self.ref_label_paths[index]).convert('L').resize((256, 256), Image.NEAREST)
        ref_label = self.transform_label(ref_label)*255
        ref_img = Image.open(self.ref_img_paths[index]).convert('RGB')
        ref_img = self.transform_img(ref_img)
        return label, img, ref_label, ref_img

    def __len__(self):
        return len(self.label_paths)

def cal_feat_dist(feat_1, feat_2, label_1, label_2, use_cos):
    label_u1 = torch.unique(label_1)
    label_u2 = torch.unique(label_2)
    union_label = [it for it in label_u1 if (it in label_u2 and it > 0)]
    total_num = 0
    for k in range(len(union_label)):
        total_num += (label_1 == union_label[k]).sum().cpu().item()
    theta_value = 0
    for k in range(len(union_label)):
        num_1 = (label_1 == union_label[k]).sum().cpu().item()
        mask = torch.zeros_like(label_1)
        mask[label_1 == union_label[k]] = 1
        mean_1 = (feat_1 * mask.expand(feat_1.shape[0], -1, -1)).sum(-1).sum(-1) / num_1
        num_2 = (label_2 == union_label[k]).sum()
        mask = torch.zeros_like(label_2)
        mask[label_2 == union_label[k]] = 1
        mean_2 = (feat_2 * mask.expand(feat_2.shape[0], -1, -1)).sum(-1).sum(-1) / num_2
        cos_value = (normalize(mean_1.unsqueeze(0).cpu().numpy()) * normalize(mean_2.unsqueeze(0).cpu().numpy())).sum()
        if use_cos == 'cos':
            theta_value += cos_value * num_1 / total_num
        else:
            theta = np.arccos(cos_value) / np.pi * 180
            theta_value += theta * num_1 / total_num
    return theta_value

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = 'cuda'
    vgg = VGG19_feature_color_torchversion(vgg_normal_correct=True)
    vgg.load_state_dict(torch.load('/home/jiangchang/project/CoCosNetv3/vgg/vgg19_conv.pth'))
    vgg.cuda()
    for param in vgg.parameters():
        param.requires_grad = False
    # MUNIT /home/jiangchang/project/MUNIT/output/fake
    # adain /home/jiangchang/dataset/adain
    # blendgan /home/jiangchang/dataset/Blendoutput
    # coco /home/jiangchang/project/CoCosNet-ade20k-aahq/checkpoints/aahq/test/fake
    # mcl /home/jiangchang/project/CoCosNet-ade20k-nce_aahq/checkpoints/aahq/test/fake
    # dy /home/jiangchang/project/DynaST_aahq/checkpoints/aahq/test/fake

    fake_path = '/home/jiangchang/project/CoCosNet-v2/checkpoints/metfaces/test/fake'
    dataset_mode = 'metfaces'
    use_cos = 'cos'
    dataset = Dataset_(fake_path, dataset_mode)

    value = {'r00': [], 'r12':[], 'r22':[], 'r32':[], 'r42':[], 'r52':[]}
    layers = ['r12', 'r22', 'r32', 'r42', 'r52']
    for i in range(len(dataset)): #
        if i % 100 == 0:
            print('{} / {}'.format(i, len(dataset)))
        label, img, ref_label, ref_img = dataset[i]
        label = label.cuda()
        img = img.cuda()
        ref_label = ref_label.cuda()
        ref_img = ref_img.cuda()
        theta = cal_feat_dist(img, ref_img, label, ref_label, use_cos)
        value['r00'].append(theta)
        img_features = vgg(img.unsqueeze(0), layers, preprocess=True)
        ref_img_features = vgg(ref_img.unsqueeze(0), layers, preprocess=True)
        for j in range(len(layers)):
            img_feat = F.interpolate(img_features[j], size=[256, 256], mode='nearest').squeeze()
            ref_img_feat = F.interpolate(ref_img_features[j], size=[256, 256], mode='nearest').squeeze()
            theta = cal_feat_dist(img_feat, ref_img_feat, label, ref_label, use_cos)
            value[layers[j]].append(theta)

    for key in value.keys():
        mean = np.mean(value[key])
        print(key + ':' + str(mean))

#python ref_vgg_metric.py /mnt/blob/Output/image_translation_methods/SPADE/output/test_per_img/ade20k_vae_v100 ade20k