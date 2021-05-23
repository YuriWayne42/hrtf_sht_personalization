import torch
import torch.nn as nn
import argparse
import os
import json
import shutil
import numpy as np
import scipy.io as sio
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import random
import distutils
from dataset_ import *
from utils import *
from model_ import *
import pytorch_ssim
from lpips_pytorch import LPIPS
from pytorch_model_summary import summary

def reconLoss(input_sht, target_sht, gt_hrtf, shvec):
    mse = torch.nn.MSELoss()
    shvec = shvec.float().to(input_sht.device).unsqueeze(0).repeat(input_sht.shape[0], 1, 1)
    recon = mse(torch.bmm(shvec, input_sht), gt_hrtf)
    return recon

def mseLossPlusRecon(input_sht, target_sht, gt_hrtf, shvec):
    mse = torch.nn.MSELoss()
    mse_loss = mse(input_sht, target_sht)
    shvec = shvec.float().to(input_sht.device).unsqueeze(0).repeat(input_sht.shape[0], 1, 1)
    recon = mse(torch.bmm(shvec, input_sht), gt_hrtf)
    return mse_loss + recon

def initParams():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed', type=int, help="random number seed", default=688)

    # Data folder prepare
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=True, default='./models/try/')

    # Dataset parameters
    parser.add_argument("-a", "--anthro_mat_path", type=str, default='/data/neil/HRTF/AntrhopometricMeasures.csv')
    parser.add_argument("-t", "--hrtf_SHT_mat_path", type=str,
                        default='/data/neil/HRTF/HUTUBS_matrix_measured.mat')
    parser.add_argument("-v", "--shvec_path", type=str,
                        default='/data/neil/HRTF/SH_vec_matrix.mat')
    parser.add_argument("-i", "--val_idx", type=int, default=0, help="index for Leave-one-out validation")
    parser.add_argument("--norm_anthro", type=str2bool, nargs='?', const=True, default=True,
                        help="whether to normalize anthro measures.")
    parser.add_argument('--anthro_norm_method', type=str, default='chun2017',
                        choices=['standard', 'chun2017'],
                        help="normalization method for input anthropometric measurements")

    # Dataset prepare
    parser.add_argument("--ear_anthro_dim", type=int, help="ear anthro dimension", default=12)
    parser.add_argument("--head_anthro_dim", type=int, help="head anthro dimension", default=13)
    parser.add_argument("--freq_bin", type=int, help="number of frequency bin", default=41)

    # Training prepare
    parser.add_argument("--ear_emb_dim", type=int, help="ear embedding dimension", default=32)
    parser.add_argument("--head_emb_dim", type=int, help="head embedding dimension", default=32)
    parser.add_argument("--lr_emb_dim", type=int, help="left_or_right embedding dimension", default=16)
    parser.add_argument("--freq_emb_dim", type=int, help="frequency embedding dimension", default=16)

    parser.add_argument("--condition_dim", type=int, default=256, help="dimension of encoded conditions")

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=2000, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=1024, help="Mini batch size for training")
    parser.add_argument("--lr", type=float, default=0.0005, help="adam learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.8, help="decay learning rate")
    parser.add_argument('--interval', type=int, default=100, help="interval to decay lr")

    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="1")
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers")

    parser.add_argument('--norm', type=str, default='layer', choices=['batch', 'layer', 'instance'],
                        help="normalization method")

    parser.add_argument('--target', type=str, default='sht', choices=['sht', 'hrtf'])
    parser.add_argument('--test_only', action='store_true',
                        help="test the trained model in case the test crash sometimes or another test method")

    args = parser.parse_args()

    # Change this to specify GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set seeds
    setup_seed(args.seed)

    # Path for output data
    if args.test_only:
        pass
    else:
        if not os.path.exists(args.out_fold):
            os.makedirs(args.out_fold)
        else:
            print("The output folder has already existed, please change another folder")

        # Folder for intermediate results
        if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
            os.makedirs(os.path.join(args.out_fold, 'checkpoint'))

    # Path for input data
    assert os.path.exists(args.anthro_mat_path)
    assert os.path.exists(args.hrtf_SHT_mat_path)

    # Save training arguments
    with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
        file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))

    with open(os.path.join(args.out_fold, 'train_loss.log'), 'w') as file:
        file.write("Start recording training loss ...\n")
    with open(os.path.join(args.out_fold, 'dev_loss.log'), 'w') as file:
        file.write("Start recording validation loss ...\n")
    with open(os.path.join(args.out_fold, 'test_loss.log'), 'w') as file:
        file.write("Start recording test loss ...\n")

    if not os.path.exists(os.path.join(args.out_fold, 'pic')):
        os.makedirs(os.path.join(args.out_fold, 'pic'))

    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.shvec = torch.from_numpy(sio.loadmat(args.shvec_path)["SH_Vec_matrix"])
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args

def adjust_learning_rate(args, lr, optimizer, epoch_num):
    lr = lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)

def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)

    training_set = HUTUBS(args, val=False)
    val_set = HUTUBS(args, val=True)
    trainDataLoader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    valDataLoader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # Loss functions
    metric = torch.nn.MSELoss()

    criterion = nn.BCEWithLogitsLoss()
    model = ConvNNHrtfSht(args).to(args.device)

    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                   betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=0.0005)
    prev_loss = 1e8
    early_stop_cnt = 0

    for epoch_num in tqdm(range(args.num_epochs)):
        model.train()

        trainlossDict = defaultdict(list)
        adjust_learning_rate(args, args.lr, optimizer, epoch_num)

        print('\nEpoch: %d ' % (epoch_num + 1))

        for i, (ear_anthro, head_anthro, hrtf, sht, subject, freq, left_or_right) in enumerate(tqdm(trainDataLoader)):
            ear_anthro = ear_anthro.float().to(args.device)
            head_anthro = head_anthro.float().to(args.device)
            hrtf = hrtf.float().to(args.device)
            sht = sht.float().to(args.device)
            frequency = freq.to(args.device)
            left_or_right = left_or_right.to(args.device)

            optimizer.zero_grad()

            # Generate a batch of hrtf
            gen_sht = model(ear_anthro, head_anthro, frequency, left_or_right)

            if args.target == "hrtf":
                recon_loss = metric(gen_sht, hrtf)
            else:
                recon_loss = metric(gen_sht, sht)

            loss = recon_loss
            trainlossDict["recon"].append(recon_loss.item())
            trainlossDict["gen_l"].append(loss.item())
            loss.backward()
            optimizer.step()

            with open(os.path.join(args.out_fold, "train_loss.log"), "a") as log:
                log.write(str(epoch_num) + "\t" + str(i) + "\t"
                          + str(trainlossDict["recon"][-1]) + "\t"
                          + "\n")

        vallossDict = defaultdict(list)
        generator.eval()
        with torch.no_grad():
            for i, (ear_anthro, head_anthro, hrtf, sht, subject, freq, left_or_right) in enumerate(tqdm(valDataLoader)):
                ear_anthro = ear_anthro.float().to(args.device)
                head_anthro = head_anthro.float().to(args.device)
                hrtf = hrtf.float().to(args.device)
                sht = sht.float().to(args.device)
                frequency = freq.to(args.device)
                left_or_right = left_or_right.to(args.device)

                noise_dist = torch.distributions.normal.Normal(0, 1)
                noise = noise_dist.sample((frequency.shape[0], args.noise_dim)).to(args.device)

                gen_sht = generator(noise, ear_anthro, head_anthro, frequency, left_or_right)
                if args.target == "hrtf":
                    recon_loss = metric(gen_sht, hrtf)
                else:
                    recon_loss = metric(gen_sht, sht)

                loss = recon_loss

                vallossDict["recon"].append(recon_loss.item())
                vallossDict["gen_l"].append(loss.item())

        with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:
            log.write(str(epoch_num) + "\t"
                      + str(np.nanmean(vallossDict["recon"])) + "\t"
                      + "\n")

        trainLoss = np.mean(trainlossDict['disc_l']+trainlossDict['gen_l'])
        valLoss = np.mean(vallossDict['gen_l'])

        print('Train loss: %.5f, Val loss: %.5f' % (trainLoss, valLoss))

        if ((epoch_num + 1) % 1) == 0:
            torch.save(generator.state_dict(), os.path.join(args.out_fold, 'checkpoint', 'generator_epoch%d_train%.3f_val%.3f.pt' % (epoch_num+1, trainLoss, valLoss)))

        if valLoss < prev_loss:
            torch.save(model.state_dict(), os.path.join(args.out_fold, 'model.pt'))
            print("Model is saved to: ", os.path.join(args.out_fold, 'model.pt'))
            prev_loss = valLoss
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

def rmse(first, second):
    return torch.sqrt(torch.mean((first - second) **2))

def calLSD(gen_sht, gt_sht, gt_hrtf, shvec, use_linear):
    shvec = shvec.float().to(gen_sht.device).unsqueeze(0).repeat(gen_sht.shape[0], 1, 1)
    gen_hrtf = torch.bmm(shvec, gen_sht)
    recon_hrtf = torch.bmm(shvec, gt_sht)
    if use_linear:
        gen_hrtf = 20 * torch.log(gen_hrtf)
        gt_hrtf = 20 * torch.log(gt_hrtf)
    recon_lsd = rmse(gen_hrtf, gt_hrtf)
    recon_lsd00 = rmse(gen_hrtf[:, 202, :], gt_hrtf[:, 202, :])
    lsd_recon = rmse(gen_hrtf, recon_hrtf)
    return recon_lsd, recon_lsd00, lsd_recon

def test(args):
    eval_set = HUTUBS(args, val=True)
    evalDataLoader = DataLoader(eval_set, batch_size=args.batch_size // 32, shuffle=False)

    # Loss functions
    mse_loss = torch.nn.MSELoss().to(args.device)

    # Initialize generator and discriminator
    model = ConvNNHrtfSht(args).to(args.device)
    model.load_state_dict(
        torch.load(os.path.join(args.out_fold, 'generator.pt'), map_location="cuda" if args.cuda else "cpu"),
        strict=False)
    model.eval()

    with torch.no_grad():
        sht_array = []
        gen_sht_array = []
        hrtf_array = []

        recon_loss_array = []
        for i, (ear_anthro, head_anthro, hrtf, sht, subject, freq, left_or_right) in enumerate(
                tqdm(evalDataLoader)):
            ear_anthro = ear_anthro.float().to(args.device)
            head_anthro = head_anthro.float().to(args.device)
            hrtf = hrtf.float().to(args.device)
            sht = sht.float().to(args.device)
            frequency = freq.to(args.device)
            left_or_right = left_or_right.to(args.device)

            noise_dist = torch.distributions.normal.Normal(0, 1)
            noise = noise_dist.sample((frequency.shape[0], args.noise_dim)).to(args.device)

            gen_sht = generator(noise, ear_anthro, head_anthro, frequency, left_or_right)

            if args.target == "hrtf":
                recon_loss = mse_loss(gen_sht, hrtf)
            else:
                recon_loss = mse_loss(gen_sht, sht)

            recon_loss_array.append(recon_loss.item())

            # hrtf_recon = calLSD(gen_sht, hrtf, args.shvec)

            sht_array.append(sht.squeeze(0).cpu())
            gen_sht_array.append(gen_sht.squeeze(0).cpu())
            hrtf_array.append(hrtf.squeeze(0).cpu())
            plt.close()

    sht_array = torch.cat(sht_array)
    sht_array = torch.cat(torch.split(sht_array, [sht_array.shape[0]//2, sht_array.shape[0]//2], dim=0), dim=2)

    gen_sht_array = torch.cat(gen_sht_array)
    gen_sht_array = torch.cat(torch.split(gen_sht_array, [gen_sht_array.shape[0]//2, gen_sht_array.shape[0]//2], dim=0), dim=2)

    hrtf_array = torch.cat(hrtf_array)
    hrtf_array = torch.cat(torch.split(hrtf_array, [hrtf_array.shape[0]//2, hrtf_array.shape[0]//2], dim=0), dim=2)

    if args.target == "hrtf":
        print(rmse(gen_sht_array, hrtf_array).item())
    else:
        lsd, lsd00, lsd_recon = calLSD(gen_sht_array, sht_array, hrtf_array, args.shvec, args.use_linear)
        print("LSD:", lsd.item(), "LSD00:", lsd00.item(), "LSDrecon:", lsd_recon.item())

    sio.savemat(os.path.join(args.out_fold, "result_%02d.mat" % args.val_idx),
                {"sht_array": sht_array.numpy(), "gen_sht_array": gen_sht_array.numpy()})

if __name__ == "__main__":
    args = initParams()
    shvec = torch.from_numpy(sio.loadmat("/data/neil/HRTF/SH_vec_matrix.mat")["SH_Vec_matrix"])

    ear_anthro = torch.rand((128, 12))
    head_anthro = torch.rand((128, 13))
    frequency = torch.LongTensor(np.random.randint(0, 41, 128))
    left_or_right = torch.LongTensor(np.random.randint(0, 2, 128))
    model = ConvNNHrtfSht(args)

    output = model(ear_anthro, head_anthro, frequency, left_or_right)

    print(summary(ConvNNHrtfSht(args), ear_anthro, head_anthro, frequency, left_or_right, show_input=False))

    if not args.test_only:
        train(args)
    test(args)
