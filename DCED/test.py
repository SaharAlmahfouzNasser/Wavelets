from networks import DCED
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import dataloader
import os
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import dill
from torchvision.utils import save_image
import torch
from tqdm import tqdm
import losses
import numpy as np
from losses import MSE
from metrics import *
import dataloader

TEST_BATCH_SIZE = 1
WORKERS = 8
start_epoch = 1
DEVICE = 'cuda:0'
EPOCHS = 300
LOAD_CHECKPOINT = 'checkpoint.pth.tar'
TEST_SAVE_PATH = 'Test_results'


def NORM(x):
    Min = torch.min(x)
    Max = torch.max(x)
    y = (x-Min)/(Max-Min+1e-10)
    return y

#test_hr_path = '/home/sahar/Desktop/SuperMudi_related_stuff/Data/subject5/HR/MB_Re_t_moco_registered_applytopup.nii.gz'
#test_lr_path = '/home/sahar/Desktop/SuperMudi_related_stuff/Data/subject5/LR/MB_Re_t_moco_registered_applytopup_isotropic_voxcor.nii.gz'

test_hr_path = '/home/sahar/Desktop/SuperMudi_related_stuff/Data/subject2/HR/MB_Re_t_moco_registered_applytopup_resized.nii.gz'
test_lr_path = '/home/sahar/Desktop/SuperMudi_related_stuff/Data/subject2/LR/MB_Re_t_moco_registered_applytopup_isotropic_voxcor.nii.gz'


def pbar_desc(label, epoch, total_epochs, loss_val, losses):
    return f'{label}: {epoch:04d}/{total_epochs} | {loss_val:.3f} | mse: {losses["mse"]}'
def save_images(path, lr_images, fake_hr, hr_images, epoch, batchid):

    images_path = os.path.join(path, f'{epoch:04d}')

    if not os.path.exists(images_path):
        os.makedirs(images_path)

    for i, tensor in enumerate(lr_images):
        np.save(f'{images_path}/{batchid}_{i:02d}_lr.jpg', tensor.numpy())

    for i, tensor in enumerate(fake_hr):
        np.save(f'{images_path}/{batchid}_{i:02d}_fake.jpg', tensor.numpy())

    for i, tensor in enumerate(hr_images):
        tensor = tensor.cpu()
        np.save(f'{images_path}/{batchid}_{i:02d}_hr.jpg', tensor.numpy())


def evaluate(DCED_net, test_dl, epoch, epochs, Loss):
    # Set the nets into evaluation mode
    DCED_net.eval()
    
    if not os.path.exists(TEST_SAVE_PATH):
        os.mkdir(TEST_SAVE_PATH)
    v_pbar = tqdm(test_dl, desc=pbar_desc('test', epoch, epochs, 0.0, {'mse': 0.0}))
    with torch.no_grad():
        MSE_all = 0.0
        SSIM_all = 0.0
        PSNR_all =0.0
        Number_imgs = 0.0
        e = 0.0
        for lr_imgs, hr_imgs in v_pbar:
            e=e+1
            Number_imgs = Number_imgs + 1
            lr_imgs = NORM(lr_imgs).to(DEVICE)
            hr_imgs = NORM(hr_imgs).to(DEVICE)

            out_imgs = NORM(DCED_net(lr_imgs))
            mse_loss = Loss(out_imgs, hr_imgs)
            mse_display = mse_loss.detach().cpu().item() 
            MSE_all=MSE_all + mse_loss

            SSIM_all = SSIM_all + SSIM(hr_imgs[0,:,:], out_imgs[0,:,:])
            PSNR_all = PSNR_all +PSNR(hr_imgs[0,:,:], out_imgs[0,:,:])
            v_pbar.set_description(pbar_desc('test', epoch, EPOCHS, mse_loss.item(), {'mse': round(mse_display, 3)}))
            if e==10:
                # Save samples at the end
                j=10
                save_images(TEST_SAVE_PATH, lr_imgs.detach().cpu(), out_imgs.detach().cpu(), hr_imgs, epoch, j)
            if e==100:
                # Save samples at the end
                j=100
                save_images(TEST_SAVE_PATH, lr_imgs.detach().cpu(), out_imgs.detach().cpu(), hr_imgs, epoch, j)
            if e==1000:
                # Save samples at the end
                j=1000
                save_images(TEST_SAVE_PATH, lr_imgs.detach().cpu(), out_imgs.detach().cpu(), hr_imgs, epoch, j)
        mse_all_avg = MSE_all/Number_imgs
        ssim_all_avg = SSIM_all/Number_imgs
        psnr_all_avg = PSNR_all/Number_imgs
    return mse_all_avg, ssim_all_avg,psnr_all_avg



def main():
    
    print('Loading Data ....')
    test_ds = dataloader.MRI_Dataset(test_lr_path, test_hr_path)
    test_dl = DataLoader(test_ds, TEST_BATCH_SIZE, shuffle=False, num_workers=WORKERS)
    DCED_net = DCED(1,1)
    if LOAD_CHECKPOINT is not None:
        checkpoint = torch.load(LOAD_CHECKPOINT, pickle_module=dill)
        start_epoch = checkpoint['epoch']
        DCED_net.load_state_dict(checkpoint['DCED_net_state_dict'])
        opt = checkpoint['optimizer']
        sched = checkpoint['lr_scheduler']

    DCED_net.to(DEVICE)
    # Losses
    Loss = MSE()
    Loss.to(DEVICE)
    epoch = 1
    mse_all_avg,ssim_all_avg,psnr_all_avg = evaluate(DCED_net, test_dl, epoch, EPOCHS,Loss)
    print("The average mse of this subject:",mse_all_avg)
    print("The average ssim of this subject:",ssim_all_avg)
    print("The average psnr of this subject:",psnr_all_avg)

if __name__ == '__main__':
    main()



