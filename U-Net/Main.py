
from networks import U_Net
from networks import init_weights
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm
import dataloader
import losses
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import dill
from torchvision.utils import save_image
from losses import MSE
import random
from metrics import *
from save import *
from train import *
from option import args

TRAIN_BATCH_SIZE = args.TRAIN_BATCH_SIZE
VAL_BATCH_SIZE = args.VAL_BATCH_SIZE
LR = args.LR
WORKERS = args.WORKERS
DEVICE = args.DEVICE
LR_DECAY = args.LR_DECAY
LR_STEP= args.LR_STEP
TRAIN_IN = args.TRAIN_IN
TRAIN_OUT = args.TRAIN_OUT
VAL_IN = args.VAL_IN
VAL_OUT = args.VAL_OUT
EXP_NO = args.EXP_NO
LOAD_CHECKPOINT = args.LOAD_CHECKPOINT 
TENSORBOARD_LOGDIR = args.TENSORBOARD_LOGDIR 
END_EPOCH_SAVE_SAMPLES_PATH = args.END_EPOCH_SAVE_SAMPLES_PATH
WEIGHTS_SAVE_PATH = args.WEIGHTS_SAVE_PATH 
LOSS_WEIGHT = args.LOSS_WEIGHT
BATCHES_TO_SAVE = args.BATCHES_TO_SAVE 
SAVE_EVERY = args.SAVE_EVERY 
VISUALIZE_EVERY = args.VISUALIZE_EVERY 
EPOCHS = args.EPOCHS
NORM = args.NORM
SCALE = args.SCALE
def main():
    transforms = T.Compose([T.ToTensor()])#, T.CenterCrop(64)])
    trn_ds = dataloader.MRI_Dataset(TRAIN_IN, TRAIN_OUT,transforms)
    trn_dl = DataLoader(trn_ds, TRAIN_BATCH_SIZE, shuffle = True, num_workers = WORKERS)
    val_ds = dataloader.MRI_Dataset(VAL_IN, VAL_OUT,transforms)
    val_dl = DataLoader(val_ds, VAL_BATCH_SIZE, shuffle = True, num_workers = WORKERS)
    start_epoch = 1
    best_val_loss = float('inf')

    DCED_net = U_Net(4,4)
    print(DCED_net)
    print('DCED parameters:', sum(p.numel() for p in DCED_net.parameters()))
    init_weights(DCED_net)
    opt = optim.Adam(DCED_net.parameters(), lr = LR)
    sched = optim.lr_scheduler.StepLR(opt, LR_STEP, gamma=LR_DECAY)

    if not os.path.exists(WEIGHTS_SAVE_PATH):
        os.mkdir(WEIGHTS_SAVE_PATH)

    if LOAD_CHECKPOINT is not None:
        checkpoint = torch.load(LOAD_CHECKPOINT, pickle_module = dill)
        start_epoch = checkpoint['epoch']
        DCED_net.load_state_dict(checkpoint['DCED_net_state_dict'])
        opt = checkpoint['optimizer']
        sched = checkpoint['lr_scheduler']

    DCED_net.to(DEVICE)

    Loss =MSE()#nn.BCEWithLogitsLoss()
    Loss.to(DEVICE)

    train_losses = Bookkeeping(TENSORBOARD_LOGDIR, suffix='trn')
    val_losses = Bookkeeping(TENSORBOARD_LOGDIR, suffix='val')

    for epoch in range(start_epoch, EPOCHS+1):
        ## training loop
        train(DCED_net, trn_dl,epoch,EPOCHS,Loss,opt,train_losses,DEVICE,TENSORBOARD_LOGDIR,LOSS_WEIGHT)

        ## validation loop
        best_val_loss = evaluate(DCED_net, val_dl, epoch, EPOCHS, Loss, val_losses, best_val_loss,DEVICE,TENSORBOARD_LOGDIR,END_EPOCH_SAVE_SAMPLES_PATH,WEIGHTS_SAVE_PATH,VISUALIZE_EVERY,SAVE_EVERY,LOSS_WEIGHT,EXP_NO)
        sched.step()

        save_checkpoint(epoch, DCED_net, None, opt, sched, )

        train_losses.update_tensorboard(epoch)
        val_losses.update_tensorboard(epoch)

        ## Reset all losses for the new epoch
        train_losses.reset()
        val_losses.reset()


if __name__=='__main__':
    
    main()

