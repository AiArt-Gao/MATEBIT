import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from PIL import Image

from torchvision.models.inception import inception_v3
import os
import sys
import numpy as np
from scipy.stats import entropy

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__':
    import torchvision.transforms as transforms
    
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, folder):
            self.folder = folder
            files = os.listdir(folder)
            self.imgs = [it for it in files if (it.endswith('.jpg') or it.endswith('.png'))]
            self.transform = transforms.Compose([transforms.Resize((256, 256)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                                        (0.5, 0.5, 0.5))])

        def __getitem__(self, index):
            img_path = self.imgs[index]
            img = Image.open(os.path.join(self.folder, img_path))
            img = self.transform(img)
            return img

        def __len__(self):
            return len(self.imgs)


    data_ = IgnoreLabelDataset(sys.argv[1])
    print('datalen is {}'.format(len(data_)))

    print ("Calculating Inception Score...")
    print (inception_score(data_, cuda=True, batch_size=32, resize=True, splits=10))

#python inception_score.py /mnt/blob/Output/SPADE_Exemplar/output/test_per_img/ade20k_exemplar_stage3_InoiseCwarpmask_perc0.01_attn_baseline2
#python fid_score.py /mnt/blob/Output/SPADE_Exemplar/output/test_per_img/ade20k_exemplar_stage3_InoiseCwarpmask_perc0.01_attn_baseline2 /mnt/blob/Dataset/ADEChallengeData2016/fid_256 --gpu 0