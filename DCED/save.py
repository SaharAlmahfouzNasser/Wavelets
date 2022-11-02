import torch
import numpy as np
from tensorboardX import SummaryWriter
import os
import dill
from torchvision.utils import save_image
# keep tracking the losses
class Bookkeeping:
    def __init__(self, tensorboard_log_path=None, suffix=''):
        self.loss_names = ['loss','psnr','ssim']
        self.genesis()
        ## initialize tensorboard objects
        self.tboard = dict()
        if tensorboard_log_path is not None:
            if not os.path.exists(tensorboard_log_path):
                os.mkdir(tensorboard_log_path)
            for name in self.loss_names:
                self.tboard[name] = SummaryWriter(os.path.join(tensorboard_log_path, name + '_' + suffix))

    def genesis(self):
        self.losses = {key: 0 for key in self.loss_names}
        self.count = 0

    def update(self, **kwargs):
        for key in kwargs:
            self.losses[key]+=kwargs[key]
        self.count +=1

    def reset(self):
        self.genesis()

    def get_avg_losses(self):
        avg_losses = dict()
        for key in self.loss_names:
            avg_losses[key] = self.losses[key] / self.count
        return avg_losses

    def update_tensorboard(self, epoch):
        avg_losses = self.get_avg_losses()
        for key in self.loss_names:
            self.tboard[key].add_scalar(key, avg_losses[key], epoch)

# Save the results

def save_checkpoint(epoch, DCED_net, best_metrics, optimizer, lr_scheduler, filename='checkpoint.pth.tar'):
    state = {'epoch': epoch, 'DCED_net_state_dict': DCED_net.state_dict(),
             'best_metrics': best_metrics, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
    torch.save(state, filename, pickle_module=dill)


def save_images(path, in_images,  pred_images, out_images, epoch, batchid):

    images_path = os.path.join(path, f'{epoch:04d}')

    if not os.path.exists(images_path):
        os.makedirs(images_path)

    for i, tensor in enumerate(in_images):
        #np.save(f'{images_path}/{batchid}_{i:02d}_in.jpg', tensor)
        save_image(tensor, f'{images_path}/{batchid}_{i:02d}_in.jpg')
    for i, tensor in enumerate(pred_images):
        tensor_pred = tensor.cpu()
        #np.save(f'{images_path}/{batchid}_{i:02d}_pred.jpg', tensor_pred)
        save_image(tensor_pred,f'{images_path}/{batchid}_{i:02d}_pred.jpg')
    for i, tensor in enumerate(out_images):
        tensor_out = tensor.cpu()
        #np.save(f'{images_path}/{batchid}_{i:02d}_GT.jpg', tensor_out)
        save_image(tensor_out,f'{images_path}/{batchid}_{i:02d}_out.jpg')

