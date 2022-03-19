import os
from abc import ABC

import numpy as np
from collections import OrderedDict
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
from scipy.special import entr
import pdb

from networks import get_generator
from networks.networks import gaussian_weights_init
from models.utils import AverageMeter, get_scheduler, psnr, mse, nmse, nmae
from skimage.metrics import structural_similarity as ssim


class CNNModel(nn.Module, ABC):
    def __init__(self, opts):
        super(CNNModel, self).__init__()

        self.loss_names = []  
        self.networks = []  
        self.optimizers = []  

        loss_flags = ["w_img_L1"]
        for flag in loss_flags:
            if not hasattr(opts, flag): setattr(opts, flag, 0)

        self.is_train = True if hasattr(opts, 'lr') else False

        self.net_G = get_generator(opts.net_G, opts)
        self.networks.append(self.net_G)

        if self.is_train:
            self.loss_names += ['loss_G_L1']
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(),
                                                lr=opts.lr,
                                                betas=(opts.beta1, opts.beta2),
                                                weight_decay=opts.weight_decay)
            self.optimizers.append(self.optimizer_G)

        self.criterion = nn.L1Loss()  # L1 loss function

        self.opts = opts

    def setgpu(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) 

    def initialize(self):
        [net.apply(gaussian_weights_init) for net in self.networks]

    def set_scheduler(self, opts, epoch=-1):
        self.schedulers = [get_scheduler(optimizer, opts, last_epoch=epoch) for optimizer in self.optimizers]

    def set_input(self, data):
        self.vol_NC = data['vol_NC'].to(self.device).float()
        self.vol_AC = data['vol_AC'].to(self.device).float()
        self.vol_SC = data['vol_SC'].to(self.device).float()
        self.vol_SC2 = data['vol_SC2'].to(self.device).float()
        self.vol_SC3 = data['vol_SC3'].to(self.device).float()
        self.vol_GD = data['vol_GD'].to(self.device).float()
        self.vol_BMI = data['vol_BMI'].to(self.device).float()
        self.vol_STATE = data['vol_STATE'].to(self.device).float()

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, name))
        return errors_ret

    def set_epoch(self, epoch):
        self.curr_epoch = epoch

    def forward(self):
        inp = self.vol_NC

        if self.opts.use_state:
            inp = torch.cat([inp, self.vol_STATE], 1)

        if self.opts.use_scatter:
            inp = torch.cat([inp, self.vol_SC], 1)

        if self.opts.use_scatter2:
            inp = torch.cat([inp, self.vol_SC2], 1)

        if self.opts.use_scatter3:
            inp = torch.cat([inp, self.vol_SC3], 1)

        if self.opts.use_bmi:
            inp = torch.cat([inp, self.vol_BMI], 1)

        if self.opts.use_gender:
            inp = torch.cat([inp, self.vol_GD], 1)

        inp.requires_grad_(True)  
        self.vol_AC_pred = self.net_G(inp)  


    def update_G(self):
        self.optimizer_G.zero_grad()  

        loss_G_L1 = self.criterion(self.vol_AC_pred, self.vol_AC)

        self.loss_G_L1 = loss_G_L1.item()  

        total_loss = loss_G_L1
        total_loss.backward()
        self.optimizer_G.step()

    def optimize(self):  
        self.loss_G_L1 = 0

        self.forward()
        self.update_G()

    @property  
    def loss_summary(self):
        message = ''
        if self.opts.wr_L1 > 0:
            message += 'G_L1: {:.4e} '.format(self.loss_G_L1)

        return message

    # learning rate decay
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()  # learning rate update

    def save(self, filename, epoch, total_iter):  
        state = {}  # dict
        if self.opts.wr_L1 > 0:
            state['net_G'] = self.net_G.module.state_dict()
            state['opt_G'] = self.optimizer_G.state_dict()

        state['epoch'] = epoch
        state['total_iter'] = total_iter

        torch.save(state, filename)

        print('Saved {}'.format(filename))


    def resume(self, checkpoint_file, train=True):
        checkpoint = torch.load(checkpoint_file)

        if self.opts.wr_L1 > 0:
            self.net_G.module.load_state_dict(checkpoint['net_G'])
            if train:
                self.optimizer_G.load_state_dict(checkpoint['opt_G'])

        print('Loaded {}'.format(checkpoint_file))

        return checkpoint['epoch'], checkpoint['total_iter']

    def evaluate(self, loader):
        val_bar = tqdm(loader)
        avg_psnr_AC = AverageMeter()
        avg_ssim_AC = AverageMeter()
        avg_mse_AC = AverageMeter()
        avg_nmse_AC = AverageMeter()
        avg_nmae_AC = AverageMeter()

        pred_AC_images = []
        gt_AC_images = []
        gt_NC_images = []
        gt_SC_images = []
        gt_GD_images = []
        gt_BMI_images = []

        for data in val_bar:
            self.set_input(data)
            self.forward()

            # Add the self.vol_AC normalization here
            if self.opts.norm_pred_AC:
                self.vol_AC_pred = self.vol_AC_pred/(self.vol_AC_pred.mean().item())   

            if self.opts.wr_L1 > 0:
                psnr_AC = psnr(self.vol_AC_pred, self.vol_AC)
                mse_AC =   mse(self.vol_AC_pred, self.vol_AC)
                ssim_AC = ssim(self.vol_AC_pred[0,0,...].cpu().numpy(), self.vol_AC[0,0,...].cpu().numpy())
                nmse_AC = nmse(self.vol_AC_pred, self.vol_AC)
                nmae_AC = nmae(self.vol_AC_pred, self.vol_AC)


                avg_psnr_AC.update(psnr_AC)
                avg_mse_AC.update(mse_AC)
                avg_ssim_AC.update(ssim_AC)
                avg_nmse_AC.update(nmse_AC)
                avg_nmae_AC.update(nmae_AC)

                pred_AC_images.append(self.vol_AC_pred[0].cpu())
                gt_AC_images.append(self.vol_AC[0].cpu())
                gt_NC_images.append(self.vol_NC[0].cpu())
                gt_SC_images.append(self.vol_SC[0].cpu())
                gt_GD_images.append(self.vol_GD[0].cpu())
                gt_BMI_images.append(self.vol_BMI[0].cpu())

            # Just show NMSE, NMAE, SSIM here
            message = 'NMSE: {:4f} '.format(avg_nmse_AC.avg)
            message += 'NMAE: {:4f} '.format(avg_nmae_AC.avg)
            message += 'SSIM: {:4f} '.format(avg_ssim_AC.avg)
            message += 'PSNR: {:4f} '.format(avg_psnr_AC.avg)
            val_bar.set_description(desc=message)

        self.nmse_AC = avg_nmse_AC.avg
        self.nmae_AC = avg_nmae_AC.avg
        self.ssim_AC = avg_ssim_AC.avg  
        self.psnr_AC = avg_psnr_AC.avg
        self.mse_AC = avg_mse_AC.avg


        self.results = {}  
        if self.opts.wr_L1 > 0:
            self.results['pred_AC'] = torch.stack(pred_AC_images).squeeze().numpy()
            self.results['gt_AC'] = torch.stack(gt_AC_images).squeeze().numpy()
            self.results['gt_NC'] = torch.stack(gt_NC_images).squeeze().numpy()
            self.results['gt_SC'] = torch.stack(gt_SC_images).squeeze().numpy()
            self.results['gt_GD'] = torch.stack(gt_GD_images).squeeze().numpy()
            self.results['gt_BMI'] = torch.stack(gt_BMI_images).squeeze().numpy()


