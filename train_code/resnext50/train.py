from Dataset import *
from Network import *
from Functions import *
import os
from fastai.distributed import *
import argparse
try:
    #from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0,1',  help='which gpu to use')
    parser.add_argument('--path', type=str, default='../../data', help='path of csv file with DNA sequences and labels')
    parser.add_argument('--epochs', type=int, default=32, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='size of each batch during training')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight dacay used in optimizer')
    parser.add_argument('--save_freq', type=int, default=1, help='saving checkpoints per save_freq epochs')
    parser.add_argument('--dropout', type=float, default=.1, help='transformer dropout')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--nfolds', type=int, default=4, help='number of cross validation folds')
    parser.add_argument('--fold', type=int, default=0, help='which fold to train')
    parser.add_argument('--val_freq', type=int, default=1, help='which fold to train')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--expansion', type=int, default=64, help='number of expansion pixels')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient_accumulation_steps')
    opts = parser.parse_args()
    return opts


opts=get_args()

#set up gpu
os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.system('mkdir models')
os.system('mkdir logs')

#dice = Dice_th_pred(np.arange(0.2,0.7,0.01))

#datasets and dataloaders
dataset = HuBMAPDataset(path=opts.path, fold=opts.fold, nfolds=opts.nfolds, train=True, tfms=get_aug())
val_dataset = HuBMAPDataset(path=opts.path, fold=opts.fold, nfolds=opts.nfolds, train=False)
dataloader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.workers, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.workers, drop_last=True)


#model and optimizer
model = UneXt50().cuda()
#optimizer = Ranger(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
#                                           max_lr=1e-3, epochs=opts.epochs, steps_per_epoch=len(dataloader))
criterion=nn.BCEWithLogitsLoss()
opt_level = 'O1'
model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
model = nn.DataParallel(model)
#some more things
logger=CSVLogger(['epoch','train_loss','val_loss','dice_coef'],f"logs/log_fold{opts.fold}.csv")
metric=Dice_soft()
best_metric=0

#training


scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.2, div_factor=1e2, max_lr=1e-4, epochs=opts.epochs, steps_per_epoch=len(dataloader))

for epoch in range(opts.epochs):

    print(f"### training for epoch {epoch} ###")
    train_loss=0
    model.train(True)
    #for data in tqdm(dataloader):
    #step=0
    for data in tqdm(dataloader):
        #step+=1
        img=data['img'].to(device)
        mask=data['mask'].to(device)
        img=cutout(img)
        output=model(img)
        loss=criterion(output,mask)

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        #if step%opts.gradient_accumulation_steps==0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        train_loss+=loss.item()
        #break
    train_loss/=len(dataloader)



    print(f"### validating for epoch {epoch} ###")
    val_loss=0
    model.eval()
    metric.reset()
    with torch.no_grad():
        for data in tqdm(val_dataloader):
            if img.shape[0]%2!=0:
                img=img[:-1]
                mask=mask[:-1]
            img=data['img'].to(device)
            mask=data['mask'].to(device)
            shape=img.shape
            output=model(img)#[:,:,opts.expansion//2:-opts.expansion//2,opts.expansion//2:-opts.expansion//2]
            mask=mask#[:,:,opts.expansion//2:-opts.expansion//2,opts.expansion//2:-opts.expansion//2]
            metric.accumulate(output.detach(), mask)
            loss=criterion(output,mask)
            val_loss+=loss.item()
    val_loss/=len(val_dataloader)
    metric_this_epoch=metric.value
    #metric_this_epoch=val_loss
    logger.log([epoch+1,train_loss,val_loss,metric_this_epoch])
    if metric_this_epoch>best_metric:
        torch.save(model.state_dict(),f'models/fold{opts.fold}.pth')
        best_metric=metric_this_epoch
