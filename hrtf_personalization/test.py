import torch
import scipy.io as sio
import numpy as np
from tqdm import tqdm, trange
import random
import os

for j in range(93):
    os.system("python train.py -o /data2/neil/HRTF_GAN/models/hrtf_sht%02d --gpu %d -i %d" %(j, (2 + j % 2), j))

shvec = torch.from_numpy(sio.loadmat("/data/neil/HRTF/SH_vec_matrix.mat")["SH_Vec_matrix"])
shvec = shvec.float().unsqueeze(0).repeat(41, 1, 1)

model_dir = "/data2/neil/HRTF_GAN/models/hrtf_sht"
hrtf_SHT_mat_path = '../sht_preprocessing/HUTUBS_matrix_measured.mat'

def mse(first, second):
    return (first - second) **2

all_res = []
for i in range(93):
    result_mat_path = os.path.join(model_dir + "%02d" % i, "result_%02d.mat" % i)
    gt_sht = torch.from_numpy(sio.loadmat(result_mat_path)["sht_array"])
    gen_sht = torch.from_numpy(sio.loadmat(result_mat_path)["gen_sht_array"])
    predicted_hrtf = torch.bmm(shvec, gen_sht)
    smoothed_hrtf = torch.bmm(shvec, gt_sht)
    actual_hrtf = torch.from_numpy(sio.loadmat(hrtf_SHT_mat_path)["hrtf_freq_allDB"])[i]
    all_res.append(calLSD(predicted_hrtf, smoothed_hrtf, actual_hrtf))

recon_lsd_lst, recon_lsd00_lst, lsd_recon_lst = [], [], []
for (recon_lsd, recon_lsd00, lsd_recon) in all_res:
    recon_lsd_lst.append(recon_lsd)
    recon_lsd00_lst.append(recon_lsd00)
    lsd_recon_lst.append(lsd_recon)
recon_lsd_lst_ = torch.stack(recon_lsd_lst)
recon_lsd00_lst_ = torch.stack(recon_lsd00_lst)
lsd_recon_lst_ = torch.stack(lsd_recon_lst)

print(np.sqrt(torch.mean(lsd_recon_lst_)))

print(np.sqrt(torch.mean(recon_lsd_lst_)))
