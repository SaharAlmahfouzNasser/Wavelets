import argparse

EXP_NO = 13
parser = argparse.ArgumentParser(description='Select the values of the hyper-parameters')
parser.add_argument('-TRAIN_BATCH_SIZE', default = 1)
parser.add_argument('-VAL_BATCH_SIZE', default = 1)
parser.add_argument('-LR', default = 0.0001)
parser.add_argument('-WORKERS', default = 0)
parser.add_argument('-DEVICE', default = 'cuda:2')
parser.add_argument('-LR_DECAY', default = 0.5)
parser.add_argument('-LR_STEP', default = 10)
parser.add_argument('-TRAIN_IN', default = '/home/sahar/Desktop/SuperMudi_related_stuff/Data/subject1/LR/MB_Re_t_moco_registered_applytopup_isotropic_voxcor.nii.gz')
parser.add_argument('-TRAIN_OUT', default = '/home/sahar/Desktop/SuperMudi_related_stuff/Data/subject1/HR/MB_Re_t_moco_registered_applytopup_resized.nii.gz')
parser.add_argument('-VAL_IN', default = '/home/sahar/Desktop/SuperMudi_related_stuff/Data/subject2/LR/MB_Re_t_moco_registered_applytopup_isotropic_voxcor.nii.gz')
parser.add_argument('-VAL_OUT', default =  '/home/sahar/Desktop/SuperMudi_related_stuff/Data/subject2/HR/MB_Re_t_moco_registered_applytopup_resized.nii.gz')

#parser.add_argument('-TRAIN_IN', default = '/home/sahar/wavelet/Data/train/noisy/')
#parser.add_argument('-TRAIN_OUT', default = '/home/sahar/wavelet/Data/train/clean/')
#parser.add_argument('-VAL_IN', default = '/home/sahar/wavelet/Data/val/noisy/')
#parser.add_argument('-VAL_OUT', default =  '/home/sahar/wavelet/Data/val/clean/')
parser.add_argument('-EXP_NO', default = EXP_NO)
parser.add_argument('-LOAD_CHECKPOINT', default = None)
parser.add_argument('-TENSORBOARD_LOGDIR', default = f'{EXP_NO:02d}-tboard')
parser.add_argument('-END_EPOCH_SAVE_SAMPLES_PATH', default = f'{EXP_NO:02d}-epoch_end_samples')
parser.add_argument('-WEIGHTS_SAVE_PATH', default = f'{EXP_NO:02d}-weights')
parser.add_argument('-LOSS_WEIGHT', default = 1.0)
parser.add_argument('-BATCHES_TO_SAVE', default = 1)
parser.add_argument('-SAVE_EVERY', default = 10)
parser.add_argument('-VISUALIZE_EVERY', default = 2)
parser.add_argument('-EPOCHS', default = 300)
parser.add_argument('-NORM', default = False)
parser.add_argument('-SCALE', default = 2)

args = parser.parse_args()

