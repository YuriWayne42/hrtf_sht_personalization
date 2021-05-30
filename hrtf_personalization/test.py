import torch
import scipy.io as sio
import numpy as np
from tqdm import tqdm, trange
import random
import os

num_of_subjects = 93

for j in tqdm(range(num_of_subjects)):
    os.system("python train.py -o /data2/neil/HRTF_AES/models/hrtf_sht%02d -i %d --gpu %d" %(j, j, (2 + j % 2)))

model_dir = "/data2/neil/HRTF_AES/models/hrtf_sht"
hrtf_SHT_mat_path = "../sht_preprocessing/HUTUBS_matrix_measured.mat"
shvec_path = "../sht_preprocessing/SH_vec_matrix.mat"
freqind = torch.from_numpy(sio.loadmat(hrtf_SHT_mat_path)["freq_logind"].squeeze(0))
shvec = torch.from_numpy(sio.loadmat(shvec_path)["SH_Vec_matrix"])
shvec = shvec.float().unsqueeze(0).repeat(freqind.shape[0], 1, 1)

def mse(first, second):
    return (first.float() - second.float()) **2

def calLSD(predicted_hrtf, smoothed_hrtf, actual_hrtf):
    # compare with actual hrtf
    recon_lsd = mse(predicted_hrtf, actual_hrtf)
    # frontal direction, compare with actual
    recon_lsd00 = mse(predicted_hrtf[:, 202, :], actual_hrtf[:, 202, :])
    # compare with SHT reconstructed
    lsd_recon = mse(predicted_hrtf, smoothed_hrtf)
    return recon_lsd, recon_lsd00, lsd_recon

all_res = []
for i in tqdm(range(num_of_subjects)):
    result_mat_path = os.path.join(model_dir + "%02d" % i, "result_%02d.mat" % i)
    gt_sht = torch.from_numpy(sio.loadmat(result_mat_path)["sht_array"])
    gen_sht = torch.from_numpy(sio.loadmat(result_mat_path)["gen_sht_array"])
    predicted_hrtf = torch.bmm(shvec, gen_sht)
    smoothed_hrtf = torch.bmm(shvec, gt_sht)
    actual_hrtf = torch.from_numpy(sio.loadmat(hrtf_SHT_mat_path)["hrtf_freq_allDB"][:, freqind, ...])[i]
    all_res.append(calLSD(predicted_hrtf, smoothed_hrtf, actual_hrtf))

recon_lsd_lst, recon_lsd00_lst, lsd_recon_lst = [], [], []
for (recon_lsd, recon_lsd00, lsd_recon) in all_res:
    recon_lsd_lst.append(recon_lsd)
    recon_lsd00_lst.append(recon_lsd00)
    lsd_recon_lst.append(lsd_recon)
recon_lsd_lst_ = torch.stack(recon_lsd_lst)
recon_lsd00_lst_ = torch.stack(recon_lsd00_lst)
lsd_recon_lst_ = torch.stack(lsd_recon_lst)

print(np.sqrt(np.mean(np.array(lsd_recon_lst_))))

print(np.sqrt(np.mean(np.array(recon_lsd_lst_))))
