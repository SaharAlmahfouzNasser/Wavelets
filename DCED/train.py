
import torch
from tqdm import tqdm
import losses
import numpy as np
from losses import MSE
from metrics import *
from save import *

def NORM(x):
    Min = torch.min(x)
    Max = torch.max(x)
    y = (x-Min)/(Max-Min+1e-10)
    return y

def pbar_desc(label, epoch, total_epochs,loss, psnr,ssim):
    return f'{label}:{epoch:04d}/{total_epochs} | {loss: .3f} | {psnr: .3f} | {ssim: .3f}'

def train(DCED_net,trn_dl,epoch,epochs,Loss,opt,train_losses,DEVICE,TENSORBOARD_LOGDIR,LOSS_WEIGHT):
    DCED_net.train()
    t_pbar = tqdm(trn_dl, desc=pbar_desc('train',epoch,epochs,0.0,0.0,0.0))
    for in_imgs, out_imgs in t_pbar:
        in_imgs = NORM(in_imgs).to(DEVICE)
        out_imgs = NORM(out_imgs).to(DEVICE)
             
        pred_imgs = NORM(DCED_net(in_imgs))
        loss = Loss(pred_imgs,out_imgs)
        loss_display = loss.detach().cpu().item()

        loss = LOSS_WEIGHT * loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        psnr = PSNR(out_imgs,pred_imgs)
        ssim = SSIM(out_imgs,pred_imgs)
        t_pbar.set_description(pbar_desc('train',epoch,epochs,loss.item(), psnr,ssim))

        train_losses.update(loss = loss.item(), psnr = psnr, ssim = ssim)


def evaluate(DCED_net,val_dl,epoch,epochs,Loss,val_losses,best_val_loss,DEVICE,TENSORBOARD_LOGDIR,END_EPOCH_SAVE_SAMPLES_PATH,WEIGHTS_SAVE_PATH,VISUALIZE_EVERY,SAVE_EVERY,LOSS_WEIGHT,EXP_NO):
    DCED_net.eval()
    v_pbar = tqdm(val_dl,desc=pbar_desc('valid', epoch , epochs, 0.0, 0.0,0.0))
    with torch.no_grad():
        e = 0.0
        for in_imgs,out_imgs in v_pbar:
            e = e+1
            in_imgs = NORM(in_imgs).to(DEVICE)
            out_imgs = NORM(out_imgs).to(DEVICE)
            pred_imgs = NORM(DCED_net(in_imgs))
            
            loss = Loss(out_imgs,pred_imgs)
            

            loss_display = loss.detach().cpu().item()

            loss = LOSS_WEIGHT * loss
            psnr = PSNR(out_imgs,pred_imgs)
            ssim = SSIM(out_imgs,pred_imgs)
            val_losses.update(loss = loss.item(),psnr = psnr, ssim= ssim)
            
            v_pbar.set_description(pbar_desc('valid',epoch, epochs, loss.item(), psnr,ssim))
            if e ==VISUALIZE_EVERY:
                save_images(END_EPOCH_SAVE_SAMPLES_PATH, in_imgs.detach().cpu(), pred_imgs.detach().cpu(), out_imgs, epoch, e)
    ## save best model weights
    avg_val_losses = val_losses.get_avg_losses()
    avg_val_loss = avg_val_losses['loss']
    if avg_val_loss < best_val_loss or epoch % SAVE_EVERY == 0:
        best_val_loss = loss.item()
        torch.save(DCED_net.state_dict(), f'{WEIGHTS_SAVE_PATH}/{EXP_NO:02d}-epoch-{epoch:04d}_tota-loss-{avg_val_loss:.3f}.pth')
    return best_val_loss

