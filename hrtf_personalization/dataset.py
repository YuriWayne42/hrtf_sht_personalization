import scipy.io as sio
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class HUTUBS(Dataset):
    def __init__(self, args, val):
        super(HUTUBS, self).__init__()
        valid_hrtf_index = list(range(0, 17)) + list(range(18,78)) + list(range(79,91)) + list(range(92,96))
        anthro = pd.read_csv(args.anthro_mat_path)
        self.val = val
        self.norm_anthro = args.norm_anthro
        self.anthro_mat = np.array(anthro)[np.array(valid_hrtf_index), 1:].astype(np.float64)
        self.anthro_mat_val = self.anthro_mat[[args.val_idx]]
        self.anthro_mat_train = np.delete(self.anthro_mat, args.val_idx, axis=0)

        if self.norm_anthro:
            anthro_avg = np.mean(self.anthro_mat_train, axis=0)
            anthro_std = np.std(self.anthro_mat_train, axis=0)
            self.anthro_mat_train = self.normalize(args.anthro_norm_method, self.anthro_mat_train, anthro_avg, anthro_std)
            self.anthro_mat_val = self.normalize(args.anthro_norm_method, self.anthro_mat_val, anthro_avg, anthro_std)

        self.anthro_mat_X_train = self.anthro_mat_train[:, :13]
        self.anthro_mat_D_L_train = self.anthro_mat_train[:, 13:25]
        self.anthro_mat_D_R_train = self.anthro_mat_train[:, 25:]

        self.anthro_mat_X_val = self.anthro_mat_val[:, :13]
        self.anthro_mat_D_L_val = self.anthro_mat_val[:, 13:25]
        self.anthro_mat_D_R_val = self.anthro_mat_train[:, 25:]
        hrtf_mat = np.expand_dims(sio.loadmat(args.hrtf_SHT_mat_path)["hrtf_freq_allDB"], -2)
        self.hrtf_mat = hrtf_mat[valid_hrtf_index]

        self.hrtf_mat_val = self.hrtf_mat[[args.val_idx]]
        self.hrtf_mat_train = np.delete(self.hrtf_mat, args.val_idx, axis=0)
        sht_mat = np.expand_dims(sio.loadmat(args.hrtf_SHT_mat_path)["hrtf_SHT_dBmat"], -2)

        self.sht_mat = sht_mat[valid_hrtf_index]

        self.sht_mat_val = self.sht_mat[[args.val_idx]]
        self.sht_mat_train = np.delete(self.sht_mat, args.val_idx, axis=0)

    def normalize(self, norm_method, anthro, avg, std):
        if norm_method == "standard":
            return (anthro - avg) / std
        elif norm_method == "chun2017":
            return np.reciprocal(1 + np.exp((anthro - avg) / std))
        else:
            raise ValueError("anthropometric normalization method not recognized")

    def __len__(self):
        if self.val:
            return self.hrtf_mat_val.shape[0]*self.hrtf_mat_val.shape[1]*2
        else:
            return self.hrtf_mat_train.shape[0]*self.hrtf_mat_train.shape[1]*2

    def __getitem__(self, idx):
        if self.val:
            left_or_right = idx // (self.anthro_mat_X_val.shape[0]*self.hrtf_mat_val.shape[1])
            new_idx = idx % (self.anthro_mat_X_val.shape[0]*self.hrtf_mat_val.shape[1])
            freq = new_idx // self.anthro_mat_X_val.shape[0]
            subject = new_idx % self.anthro_mat_X_val.shape[0]
            if left_or_right == 0:
                ear_anthro = self.anthro_mat_D_L_val[subject]
            else:
                ear_anthro = self.anthro_mat_D_R_val[subject]
            head_anthro = self.anthro_mat_X_val[subject]
            hrtf = self.hrtf_mat_val[subject, freq, :, :, left_or_right]
            sht = self.sht_mat_val[subject, freq, :, :, left_or_right]
        else:
            left_or_right = idx // (self.anthro_mat_X_train.shape[0] * self.hrtf_mat_train.shape[1])
            new_idx = idx % (self.anthro_mat_X_train.shape[0] * self.hrtf_mat_train.shape[1])
            freq = new_idx // self.anthro_mat_X_train.shape[0]
            subject = new_idx % self.anthro_mat_X_train.shape[0]
            if left_or_right == 0:
                ear_anthro = self.anthro_mat_D_L_train[subject]
            else:
                ear_anthro = self.anthro_mat_D_R_train[subject]
            head_anthro = self.anthro_mat_X_train[subject]
            hrtf = self.hrtf_mat_train[subject, freq, :, :, left_or_right]
            sht = self.sht_mat_train[subject, freq, :, :, left_or_right]
        return ear_anthro, head_anthro, hrtf, sht, subject, freq, left_or_right

