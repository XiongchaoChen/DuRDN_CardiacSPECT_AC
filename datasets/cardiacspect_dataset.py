import os
import h5py
import random
import numpy as np
import pdb
import torch
import torchvision.utils as utils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.data_patch_util import *


class CardiacSPECT_Train(Dataset):
    def __init__(self, opts=None):
        self.root = opts.data_root
        self.patch_size = opts.patch_size_train
        self.n_patch = opts.n_patch_train
        self.AUG = opts.AUG

        self.data_dir = os.path.join(self.root, 'train')
        self.data_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.h5')])

        self.vol_NC_all = []
        self.vol_AC_all = []
        self.vol_SC_all = []
        self.vol_SC2_all = []
        self.vol_SC3_all =[]
        self.vol_GD_all = []
        self.vol_BMI_all = []
        self.vol_STATE_all = []

        # load all images and patching
        for filename in self.data_files:
            print('Patching: ' + str(filename))

            with h5py.File(filename, 'r') as f:
                vol_AC = f['AC'][...]  
                vol_NC = f['NC'][...]
                vol_SC = f['SC'][...]
                vol_SC2 = f['SC2'][...]
                vol_SC3 = f['SC3'][...]
                vol_GD = f['GD'][...]
                vol_BMI = f['BMI'][...]
                vol_STATE = f['STATE'][...]

            # create the random index for cropping patches
            X_template = vol_NC
            indexes = get_random_patch_indexes(data=X_template, patch_size=self.patch_size, num_patches=self.n_patch, padding='VALID')

            # use index to crop patches
            X_patches = get_patches_from_indexes(image=vol_NC, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_NC_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_AC, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_AC_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_SC, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_SC_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_SC2, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_SC2_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_SC3, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_SC3_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_GD, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_GD_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_BMI, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_BMI_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_STATE, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_STATE_all.append(X_patches)

        self.vol_NC_all = np.concatenate(self.vol_NC_all, 0) / opts.norm_NC
        self.vol_AC_all = np.concatenate(self.vol_AC_all, 0) / opts.norm_AC
        self.vol_SC_all = np.concatenate(self.vol_SC_all, 0) / opts.norm_SC
        self.vol_SC2_all = np.concatenate(self.vol_SC2_all, 0) / opts.norm_SC2
        self.vol_SC3_all = np.concatenate(self.vol_SC3_all, 0) / opts.norm_SC3
        self.vol_GD_all = np.concatenate(self.vol_GD_all, 0) / opts.norm_GD  
        self.vol_BMI_all = np.concatenate(self.vol_BMI_all, 0) / opts.norm_BMI  
        self.vol_STATE_all = np.concatenate(self.vol_STATE_all, 0) / opts.norm_STATE  

    def __getitem__(self, index):
        vol_NC = self.vol_NC_all[index, ...]
        vol_AC = self.vol_AC_all[index, ...]
        vol_SC = self.vol_SC_all[index, ...]
        vol_SC2 = self.vol_SC2_all[index, ...]
        vol_SC3 = self.vol_SC3_all[index, ...]
        vol_GD = self.vol_GD_all[index, ...]
        vol_BMI = self.vol_BMI_all[index, ...]
        vol_STATE = self.vol_STATE_all[index, ...]

        if self.AUG:  # Rotation & Augmentation
            if random.randint(0, 1):  # random rotation
                vol_NC = np.flip(vol_NC, axis=1)
                vol_AC = np.flip(vol_AC, axis=1)
                vol_SC = np.flip(vol_SC, axis=1)
                vol_SC2 = np.flip(vol_SC2, axis=1)
                vol_SC3 = np.flip(vol_SC3, axis=1)
                vol_GD = np.flip(vol_GD, axis=1)
                vol_BMI = np.flip(vol_BMI, axis=1)
                vol_STATE = np.flip(vol_STATE, axis=1)

            if random.randint(0, 1):
                vol_NC = np.flip(vol_NC, axis=2)
                vol_AC = np.flip(vol_AC, axis=2)
                vol_SC = np.flip(vol_SC, axis=2)
                vol_SC2 = np.flip(vol_SC2, axis=2)
                vol_SC3 = np.flip(vol_SC3, axis=2)
                vol_GD = np.flip(vol_GD, axis=2)
                vol_BMI = np.flip(vol_BMI, axis=2)
                vol_STATE = np.flip(vol_STATE, axis=2)

            if random.randint(0, 1):
                vol_NC = np.flip(vol_NC, axis=3)
                vol_AC = np.flip(vol_AC, axis=3)
                vol_SC = np.flip(vol_SC, axis=3)
                vol_SC2 = np.flip(vol_SC2, axis=3)
                vol_SC3 = np.flip(vol_SC3, axis=3)
                vol_GD = np.flip(vol_GD, axis=3)
                vol_BMI = np.flip(vol_BMI, axis=3)
                vol_STATE = np.flip(vol_STATE, axis=3)

            if random.randint(0, 1):
                vol_NC = np.rot90(vol_NC, axes=(1, 2))
                vol_AC = np.rot90(vol_AC, axes=(1, 2))
                vol_SC = np.rot90(vol_SC, axes=(1, 2))
                vol_SC2 = np.rot90(vol_SC2, axes=(1, 2))
                vol_SC3 = np.rot90(vol_SC3, axes=(1, 2))
                vol_GD = np.rot90(vol_GD, axes=(1, 2))
                vol_BMI = np.rot90(vol_BMI, axes=(1, 2))
                vol_STATE = np.rot90(vol_STATE, axes=(1, 2))

            if random.randint(0, 1):
                vol_NC = np.rot90(vol_NC, axes=(1, 3))
                vol_AC = np.rot90(vol_AC, axes=(1, 3))
                vol_SC = np.rot90(vol_SC, axes=(1, 3))
                vol_SC2 = np.rot90(vol_SC2, axes=(1, 3))
                vol_SC3 = np.rot90(vol_SC3, axes=(1, 3))
                vol_GD = np.rot90(vol_GD, axes=(1, 3))
                vol_BMI = np.rot90(vol_BMI, axes=(1, 3))
                vol_STATE = np.rot90(vol_STATE, axes=(1, 3))

            if random.randint(0, 1):
                vol_NC = np.rot90(vol_NC, axes=(2, 3))
                vol_AC = np.rot90(vol_AC, axes=(2, 3))
                vol_SC = np.rot90(vol_SC, axes=(2, 3))
                vol_SC2 = np.rot90(vol_SC2, axes=(2, 3))
                vol_SC3 = np.rot90(vol_SC3, axes=(2, 3))
                vol_GD = np.rot90(vol_GD, axes=(2, 3))
                vol_BMI = np.rot90(vol_BMI, axes=(2, 3))
                vol_STATE= np.rot90(vol_STATE, axes=(2, 3))

        vol_NC = torch.from_numpy(vol_NC.copy())
        vol_AC = torch.from_numpy(vol_AC.copy())
        vol_SC = torch.from_numpy(vol_SC.copy())
        vol_SC2 = torch.from_numpy(vol_SC2.copy())
        vol_SC3 = torch.from_numpy(vol_SC3.copy())
        vol_GD = torch.from_numpy(vol_GD.copy())
        vol_BMI = torch.from_numpy(vol_BMI.copy())
        vol_STATE = torch.from_numpy(vol_STATE.copy())

        return {'vol_AC': vol_AC,
                'vol_NC': vol_NC,
                'vol_SC': vol_SC,
                'vol_SC2': vol_SC2,
                'vol_SC3': vol_SC3,
                'vol_GD': vol_GD,
                'vol_BMI': vol_BMI,
                'vol_STATE': vol_STATE}

    def __len__(self):
        return self.vol_NC_all.shape[0]


class CardiacSPECT_Test(Dataset):
    def __init__(self, opts=None):
        self.root = opts.data_root
        self.patch_size = opts.patch_size_eval
        self.n_patch = opts.n_patch_eval

        self.AUG = opts.AUG

        self.data_dir = os.path.join(self.root, 'test') 
        self.data_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.h5')])

        self.vol_NC_all = []
        self.vol_AC_all = []
        self.vol_SC_all = []
        self.vol_SC2_all = []
        self.vol_SC3_all = []
        self.vol_GD_all = []
        self.vol_BMI_all = []
        self.vol_STATE_all = []

        # load all images and patching
        for filename in self.data_files:
            print('Patching: ' + str(filename))

            with h5py.File(filename, 'r') as f:
                vol_AC = f['AC'][...]
                vol_NC = f['NC'][...]
                vol_SC = f['SC'][...]
                vol_SC2 = f['SC2'][...]
                vol_SC3 = f['SC3'][...]
                vol_GD = f['GD'][...]
                vol_BMI = f['BMI'][...]
                vol_STATE = f['STATE'][...]

            # create the random index for cropping patches
            X_template = vol_NC
            indexes = get_ordered_patch_indexes(data=X_template, patch_size=self.patch_size, stride=[100000, 100000, 100000], padding='VALID')

            # use index to crop patches
            X_patches = get_patches_from_indexes(image=vol_NC, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_NC_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_AC, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_AC_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_SC, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_SC_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_SC2, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_SC2_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_SC3, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_SC3_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_GD, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_GD_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_BMI, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_BMI_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_STATE, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_STATE_all.append(X_patches)

        self.vol_NC_all = np.concatenate(self.vol_NC_all, 0) / opts.norm_NC
        self.vol_AC_all = np.concatenate(self.vol_AC_all, 0) / opts.norm_AC
        self.vol_SC_all = np.concatenate(self.vol_SC_all, 0) / opts.norm_SC
        self.vol_SC2_all = np.concatenate(self.vol_SC2_all, 0) / opts.norm_SC2
        self.vol_SC3_all = np.concatenate(self.vol_SC3_all, 0) / opts.norm_SC3
        self.vol_GD_all = np.concatenate(self.vol_GD_all, 0) / opts.norm_GD
        self.vol_BMI_all = np.concatenate(self.vol_BMI_all, 0) / opts.norm_BMI
        self.vol_STATE_all = np.concatenate(self.vol_STATE_all, 0) / opts.norm_STATE  

    def __getitem__(self, index):
        vol_NC = self.vol_NC_all[index, ...]
        vol_AC = self.vol_AC_all[index, ...]
        vol_SC = self.vol_SC_all[index, ...]
        vol_SC2 = self.vol_SC2_all[index, ...]
        vol_SC3 = self.vol_SC3_all[index, ...]
        vol_GD = self.vol_GD_all[index, ...]
        vol_BMI = self.vol_BMI_all[index, ...]
        vol_STATE = self.vol_STATE_all[index, ...]

        vol_NC = torch.from_numpy(vol_NC.copy())
        vol_AC = torch.from_numpy(vol_AC.copy())
        vol_SC = torch.from_numpy(vol_SC.copy())
        vol_SC2 = torch.from_numpy(vol_SC2.copy())
        vol_SC3 = torch.from_numpy(vol_SC3.copy())
        vol_GD = torch.from_numpy(vol_GD.copy())
        vol_BMI = torch.from_numpy(vol_BMI.copy())
        vol_STATE = torch.from_numpy(vol_STATE.copy())

        return {'vol_AC': vol_AC,
                'vol_NC': vol_NC,
                'vol_SC': vol_SC,
                'vol_SC2': vol_SC2,
                'vol_SC3': vol_SC3,
                'vol_GD': vol_GD,
                'vol_BMI': vol_BMI,
                'vol_STATE': vol_STATE}

    def __len__(self):
        return self.vol_NC_all.shape[0]


class CardiacSPECT_Valid(Dataset):
    def __init__(self, opts=None):
        self.root = opts.data_root
        self.patch_size = opts.patch_size_eval
        self.n_patch = opts.n_patch_eval

        self.AUG = opts.AUG

        self.data_dir = os.path.join(self.root, 'valid') 
        self.data_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.h5')])

        self.vol_NC_all = []
        self.vol_AC_all = []
        self.vol_SC_all = []
        self.vol_SC2_all = []
        self.vol_SC3_all = []
        self.vol_GD_all = []
        self.vol_BMI_all = []
        self.vol_STATE_all = []

        # load all images and patching
        for filename in self.data_files:
            print('Patching: ' + str(filename))

            with h5py.File(filename, 'r') as f:
                vol_AC = f['AC'][...]
                vol_NC = f['NC'][...]
                vol_SC = f['SC'][...]
                vol_SC2 = f['SC2'][...]
                vol_SC3 = f['SC3'][...]
                vol_GD = f['GD'][...]
                vol_BMI = f['BMI'][...]
                vol_STATE = f['STATE'][...]

            # create the random index for cropping patches
            X_template = vol_NC
            indexes = get_ordered_patch_indexes(data=X_template, patch_size=self.patch_size, stride=[100000, 100000, 100000], padding='VALID')

            # use index to crop patches
            X_patches = get_patches_from_indexes(image=vol_NC, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_NC_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_AC, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_AC_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_SC, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_SC_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_SC2, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_SC2_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_SC3, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_SC3_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_GD, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_GD_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_BMI, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_BMI_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_STATE, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_STATE_all.append(X_patches)

        self.vol_NC_all = np.concatenate(self.vol_NC_all, 0) / opts.norm_NC
        self.vol_AC_all = np.concatenate(self.vol_AC_all, 0) / opts.norm_AC
        self.vol_SC_all = np.concatenate(self.vol_SC_all, 0) / opts.norm_SC
        self.vol_SC2_all = np.concatenate(self.vol_SC2_all, 0) / opts.norm_SC2
        self.vol_SC3_all = np.concatenate(self.vol_SC3_all, 0) / opts.norm_SC3
        self.vol_GD_all = np.concatenate(self.vol_GD_all, 0) / opts.norm_GD
        self.vol_BMI_all = np.concatenate(self.vol_BMI_all, 0) / opts.norm_BMI
        self.vol_STATE_all = np.concatenate(self.vol_STATE_all, 0) / opts.norm_STATE  

    def __getitem__(self, index):
        vol_NC = self.vol_NC_all[index, ...]
        vol_AC = self.vol_AC_all[index, ...]
        vol_SC = self.vol_SC_all[index, ...]
        vol_SC2 = self.vol_SC2_all[index, ...]
        vol_SC3 = self.vol_SC3_all[index, ...]
        vol_GD = self.vol_GD_all[index, ...]
        vol_BMI = self.vol_BMI_all[index, ...]
        vol_STATE = self.vol_STATE_all[index, ...]

        vol_NC = torch.from_numpy(vol_NC.copy())
        vol_AC = torch.from_numpy(vol_AC.copy())
        vol_SC = torch.from_numpy(vol_SC.copy())
        vol_SC2 = torch.from_numpy(vol_SC2.copy())
        vol_SC3 = torch.from_numpy(vol_SC3.copy())
        vol_GD = torch.from_numpy(vol_GD.copy())
        vol_BMI = torch.from_numpy(vol_BMI.copy())
        vol_STATE = torch.from_numpy(vol_STATE.copy())

        return {'vol_AC': vol_AC,
                'vol_NC': vol_NC,
                'vol_SC': vol_SC,
                'vol_SC2': vol_SC2,
                'vol_SC3': vol_SC3,
                'vol_GD': vol_GD,
                'vol_BMI': vol_BMI,
                'vol_STATE': vol_STATE}

    def __len__(self):
        return self.vol_NC_all.shape[0]

if __name__ == '__main__':
    a = 1
