import os
import argparse
import json
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.backends.cudnn as cudnn

from torchvision.utils import save_image
from utils import prepare_sub_folder
from datasets import get_datasets_valid, get_datasets_test
from models import create_model

import scipy.io as sio
import csv

parser = argparse.ArgumentParser(description='CardiacSPECT')

# model name
parser.add_argument('--experiment_name', type=str, default='test_RDN_1GD_1BMI_1ST', help='give a model name before training')    # UNet_xGender_xBMI_xStage / RDN_xGender_xBMI_xStage
parser.add_argument('--model_type', type=str, default='model_cnn', help='give a model name before training')
parser.add_argument('--resume', type=str, default=None, help='Filename of the checkpoint to resume')
parser.add_argument('--resume_epoch', type=int, default=0, help='Epoch of the checkpoint to resume')

# dataset
parser.add_argument('--data_root', type=str, default='../Data/Processed/', help='data root folder')
parser.add_argument('--dataset', type=str, default='CardiacSPECT', help='dataset name')

parser.add_argument('--norm_NC', type=int, default=1, help='normalization for NC (divide by constant)')
parser.add_argument('--norm_AC', type=int, default=1, help='normalization for AC (divide by constant)')
parser.add_argument('--norm_GATE', type=int, default=1, help='normalization for GATE (divide by constant)')
parser.add_argument('--norm_SC', type=int, default=1, help='normalization for Scatter Window (divide by constant)')
parser.add_argument('--norm_SC2', type=int, default=1, help='normalization for Scatter Window2 (divide by constant)')
parser.add_argument('--norm_SC3', type=int, default=1, help='normalization for Scatter Window3 (divide by constant)')
parser.add_argument('--norm_BMI', type=int, default=40, help='normalization for BMI (divide by constant)')
parser.add_argument('--norm_GD', type=int, default=1, help='normalization for Gender (divide by constant)')
parser.add_argument('--norm_STATE', type=int, default=1, help='normalization for State (divide by constant)')
parser.add_argument('--norm_pred_AC', default=False, action='store_true', help='Mean normalization for the pred_AC images')

# network architectures, (discriminators e.g. cD, sD, are not used in the paper)
parser.add_argument('--net_G', type=str, default='RDN', help='generator network')   # UNet / RDN
parser.add_argument('--use_scatter', default=False, action='store_true', help='use scatter window information to input into the network')   # True / False
parser.add_argument('--use_scatter2', default=False, action='store_true', help='use scatter window2 information to input into the network')   # True / False
parser.add_argument('--use_scatter3', default=False, action='store_true', help='use scatter window3 information to input into the network')   # True / False
parser.add_argument('--use_gender', default=False, action='store_true', help='use gender information to input into the network')   # True / False
parser.add_argument('--use_bmi', default=False, action='store_true', help='use bmi information to input into the network')   # True / False
parser.add_argument('--use_state', default=False, action='store_true', help='use state (Stress/Rest) information to input into the network')   # True / False

parser.add_argument('--norm', type=str, default='None', help='Normalization for each convolution')  # 'BN' ,'IN', or 'None'
parser.add_argument('--norm_D', type=str, default='None', help='Normalization for each Discriminator')  # 'BN', 'IN' or 'None'

# loss options
parser.add_argument('--wr_L1', type=float, default=1, help='weight for reconstruction L1 loss')
parser.add_argument('--GAN_loss_weight', type=float, default=1, help='weight for the GAN reconstruction loss')

# training options
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epoch')
parser.add_argument('--batch_size', type=int, default=12, help='training batch size')
parser.add_argument('--n_patch_train', type=int, default=36, help='number of patch to crop for training')
parser.add_argument('--patch_size_train', nargs='+', type=int, default=[32, 32, 32], help='randomly cropped patch size for train')
parser.add_argument('--AUG', default=False, action='store_true', help='use augmentation')

# evaluation options
parser.add_argument('--eval_epochs', type=int, default=4, help='evaluation epochs')
parser.add_argument('--save_epochs', type=int, default=4, help='save evaluation for every number of epochs')
parser.add_argument('--n_patch_eval', type=int, default=1, help='number of patch to crop for evaluation')
parser.add_argument('--patch_size_eval', nargs='+', type=int, default=[48, 64, 64], help='randomly cropped patch size for evaluation')

# optimizer
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')

parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for ADAM')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for ADAM')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

# learning rate policy
parser.add_argument('--lr_policy', type=str, default='step', help='learning rate decay policy')
parser.add_argument('--step_size', type=int, default=30, help='step size for step scheduler')
parser.add_argument('--gamma', type=float, default=0.5, help='decay ratio for step scheduler')

# logger options
parser.add_argument('--snapshot_epochs', type=int, default=4, help='save model for every number of epochs')
parser.add_argument('--log_freq', type=int, default=100, help='save model for every number of epochs')
parser.add_argument('--output_path', default='./', type=str, help='Output path.')

# other
parser.add_argument('--num_workers', type=int, default=8, help='number of threads to load data')
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='list of gpu ids')
opts = parser.parse_args()

options_str = json.dumps(opts.__dict__, indent=4, sort_keys=False)
print("------------------- Options -------------------")
print(options_str[2:-2])
print("-----------------------------------------------")

cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = create_model(opts)
model.setgpu(opts.gpu_ids)

num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters: {} \n'.format(num_param))

# Resume the training model
if opts.resume is None:
    model.initialize()
    ep0 = -1
    total_iter = 0
else:
    ep0, total_iter = model.resume(opts.resume)

model.set_scheduler(opts, ep0)
ep0 += 1
print('Start training at epoch {} \n'.format(ep0))

# select dataset
train_set, val_set = get_datasets_test(opts)
train_loader = DataLoader(dataset=train_set, num_workers=opts.num_workers, batch_size=opts.batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=opts.num_workers, batch_size=1, shuffle=False)

# Setup directories
output_directory = os.path.join(opts.output_path, 'outputs', opts.experiment_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

# evaluation
print('Normal Evaluation ......')
model.eval()
with torch.no_grad():
    model.evaluate(val_loader)
sio.savemat(os.path.join(image_directory, 'eval.mat'), model.results)  

# Record the epoch, psnr, ssim and mse
with open(os.path.join(image_directory, 'epoch_psnr_ssim_mse.csv'), 'w') as f:  
    writer = csv.writer(f)
    writer.writerow(['epoch', 'NMSE', 'NMAE', 'SSIM', 'PSNR', 'MSE'])
    writer.writerow([opts.resume_epoch, model.nmse_AC, model.nmae_AC, model.ssim_AC, model.psnr_AC, model.mse_AC])

