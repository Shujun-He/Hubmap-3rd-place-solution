import torch.nn as nn
import torch.tensor as Tensor
import torch
from fastai.vision.all import *
import numpy as np

def checkpoint_avg(MODELS):
    dict=MODELS[0].module.state_dict()
    for key in dict:
        for i in range(1,len(MODELS)):
            dict[key]=dict[key]+MODELS[i].module.state_dict()[key]

        dict[key]=dict[key]/float(len(MODELS))

    MODELS[0].module.load_state_dict(dict)
    avg_model=MODELS[0]
    return avg_model

def mixup(tensor):
    shuffled_indices=torch.randperm(tensor.shape[0])
    lam=np.random.beta(1.0,1.0)
    tensor=lam*tensor+(1-lam)*tensor[shuffled_indices]
    return tensor,lam,shuffled_indices


def cutout(tensor,alpha=0.5):
    x=int(alpha*tensor.shape[2])
    y=int(alpha*tensor.shape[3])
    center=np.random.randint(0,tensor.shape[2],size=(2))
    #perm = torch.randperm(img.shape[0])
    cut_tensor=tensor.clone()
    cut_tensor[:,:,center[0]-x//2:center[0]+x//2,center[1]-y//2:center[1]+y//2]=0
    return cut_tensor

def symmetric_lovasz(outputs, targets):
    return 0.5*(lovasz_hinge(outputs, targets) + lovasz_hinge(-outputs, 1.0 - targets))

class Dice_soft(Metric):
    def __init__(self, axis=1):
        self.axis = axis
    def reset(self): self.inter,self.union = 0,0
    def accumulate(self, pred, targ):
        pred = torch.sigmoid(pred)
        self.inter += (pred*targ).float().sum().item()
        self.union += (pred+targ).float().sum().item()
    @property
    def value(self): return 2.0 * self.inter/self.union if self.union > 0 else None

# dice with automatic threshold selection
class Dice_th(Metric):
    def __init__(self, ths=np.arange(0.1,0.9,0.05), axis=1):
        self.axis = axis
        self.ths = ths

    def reset(self):
        self.inter = torch.zeros(len(self.ths))
        self.union = torch.zeros(len(self.ths))

    def accumulate(self, learn):
        pred,targ = flatten_check(torch.sigmoid(learn.pred), learn.y)
        for i,th in enumerate(self.ths):
            p = (pred > th).float()
            self.inter[i] += (p*targ).float().sum().item()
            self.union[i] += (p+targ).float().sum().item()

    @property
    def value(self):
        dices = torch.where(self.union > 0.0,
                2.0*self.inter/self.union, torch.zeros_like(self.union))
        return dices.max()

class Dice_th_pred(Metric):
    def __init__(self, ths=np.arange(0.1,0.9,0.01), axis=1):
        self.axis = axis
        self.ths = ths
        self.reset()

    def reset(self):
        self.inter = torch.zeros(len(self.ths))
        self.union = torch.zeros(len(self.ths))

    def accumulate(self,p,t):
        pred,targ = flatten_check(p, t)
        for i,th in enumerate(self.ths):
            p = (pred > th).float()
            self.inter[i] += (p*targ).float().sum().item()
            self.union[i] += (p+targ).float().sum().item()

    @property
    def value(self):
        dices = torch.where(self.union > 0.0, 2.0*self.inter/self.union,
                            torch.zeros_like(self.union))
        return dices


#iterator like wrapper that returns predicted and gt masks
class Model_pred:
    def __init__(self, model, dl, tta:bool=True, half:bool=False):
        self.model = model
        self.dl = dl
        self.tta = tta
        self.half = half

    def __iter__(self):
        self.model.eval()
        name_list = self.dl.dataset.fnames
        count=0
        with torch.no_grad():
            for x,y in iter(self.dl):
                x = x.cuda()
                if self.half: x = x.half()
                p = self.model(x)
                py = torch.sigmoid(p).detach()
                if self.tta:
                    #x,y,xy flips as TTA
                    flips = [[-1],[-2],[-2,-1]]
                    for f in flips:
                        p = self.model(torch.flip(x,f))
                        p = torch.flip(p,f)
                        py += torch.sigmoid(p).detach()
                    py /= (1+len(flips))
                if y is not None and len(y.shape)==4 and py.shape != y.shape:
                    py = F.upsample(py, size=(y.shape[-2],y.shape[-1]), mode="bilinear")
                py = py.permute(0,2,3,1).float().cpu()
                batch_size = len(py)
                for i in range(batch_size):
                    taget = y[i].detach().cpu() if y is not None else None
                    yield py[i],taget,name_list[count]
                    count += 1

    def __len__(self):
        return len(self.dl.dataset)

class Dice_th_pred(Metric):
    def __init__(self, ths=np.arange(0.1,0.9,0.01), axis=1):
        self.axis = axis
        self.ths = ths
        self.reset()

    def reset(self):
        self.inter = torch.zeros(len(self.ths))
        self.union = torch.zeros(len(self.ths))

    def accumulate(self,p,t):
        pred,targ = flatten_check(p, t)
        for i,th in enumerate(self.ths):
            p = (pred > th).float()
            self.inter[i] += (p*targ).float().sum().item()
            self.union[i] += (p+targ).float().sum().item()

    @property
    def value(self):
        dices = torch.where(self.union > 0.0, 2.0*self.inter/self.union,
                            torch.zeros_like(self.union))
        return dices

def save_img(data,name,out):
    data = data.float().cpu().numpy()
    img = cv2.imencode('.png',(data*255).astype(np.uint8))[1]
    out.writestr(name, img)

import csv
from os import path


class CSVLogger:
    def __init__(self,columns,file):
        self.columns=columns
        self.file=file
        if not self.check_header():
            self._write_header()


    def check_header(self):
        if path.exists(self.file):
            header=True
        else:
            header=False
        return header


    def _write_header(self):
        with open(self.file,"a") as f:
            string=""
            for attrib in self.columns:
                string+="{},".format(attrib)
            string=string[:len(string)-1]
            string+="\n"
            f.write(string)
        return self

    def log(self,row):
        if len(row)!=len(self.columns):
            raise Exception("Mismatch between row vector and number of columns in logger")
        with open(self.file,"a") as f:
            string=""
            for attrib in row:
                string+="{},".format(attrib)
            string=string[:len(string)-1]
            string+="\n"
            f.write(string)
        return self


class DiceCoefficient:
    def __init__(self):
        self.X=0
        self.Y=0
        self.XUY=0

    def accumulate(self,output,targets):
        predictions=output>0.5
        self.X+=predictions.sum()
        self.Y+=targets.sum()
        self.XUY+=((predictions==targets)*(targets==1)).sum()

    def reset(self):
        self.X=0
        self.Y=0
        self.XUY=0

    @property
    def value(self):
        coef=2*self.XUY/(self.X+self.Y)
        return coef

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
