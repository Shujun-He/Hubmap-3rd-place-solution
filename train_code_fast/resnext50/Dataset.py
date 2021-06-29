import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from sklearn.model_selection import KFold
import pandas as pd
import os
import torch
from albumentations import *

mean = np.array([0.65459856,0.48386562,0.69428385])
std = np.array([0.15167958,0.23584107,0.13146145])

def img2tensor(img,dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img,2)
    img = np.transpose(img,(2,0,1))
    return torch.from_numpy(img.astype(dtype, copy=False))

class HuBMAPDataset(Dataset):
    def __init__(self, path, fold=0, train=True, tfms=None, seed=2020, nfolds= 4, include_pl=False):
        self.path=path
        if include_pl:
            ids = np.concatenate([pd.read_csv(os.path.join(self.path,'train.csv')).id.values,
                     pd.read_csv(os.path.join(self.path,'sample_submission.csv')).id.values])
        else:
            ids = pd.read_csv(os.path.join(self.path,'train.csv')).id.values
        kf = KFold(n_splits=nfolds,random_state=seed,shuffle=True)
        ids = set(ids[list(kf.split(ids))[fold][0 if train else 1]])
        print(f"number of {'train512reduce4' if train else 'val'} images is {len(ids)}")
        if include_pl:
            self.fnames = ['train512reduce4/'+fname for fname in os.listdir(os.path.join(self.path,'train512reduce4')) if fname.split('_')[0] in ids]+\
            ['test/'+fname for fname in os.listdir(os.path.join(self.path,'test')) if fname.split('_')[0] in ids]
        else:
            self.fnames = ['train512reduce4/'+fname for fname in os.listdir(os.path.join(self.path,'train512reduce4')) if fname.split('_')[0] in ids]
        self.train = train
        self.tfms = tfms

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.path,fname)), cv2.COLOR_BGR2RGB)

        if self.fnames[idx][:5]=='train':
            mask = cv2.imread(os.path.join(self.path,'masks512reduce4',fname[16:]),cv2.IMREAD_GRAYSCALE)
        else:
            mask = cv2.imread(os.path.join(self.path,'test_masks',fname[5:]),cv2.IMREAD_GRAYSCALE)
        if self.tfms is not None:
            augmented = self.tfms(image=img,mask=mask)
            img,mask = augmented['image'],augmented['mask']

        data={'img':img2tensor((img/255.0 - mean)/std), 'mask':img2tensor(mask)}
        return data


def get_aug(p=1.0):
    return Compose([
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9,
                         border_mode=cv2.BORDER_REFLECT),
        OneOf([
            ElasticTransform(p=.3),
            GaussianBlur(p=.3),
            GaussNoise(p=.3),
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.3),
        OneOf([
            HueSaturationValue(15,25,0),
            CLAHE(clip_limit=2),
            RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
        ], p=0.3),
    ], p=p)
