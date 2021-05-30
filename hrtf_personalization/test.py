import os

for j in range(93):
    os.system("python train.py -o /data2/neil/HRTF_GAN/models0427/logfreq%02d --gpu %d -i %d" %(j, (j+2) % 4, j))

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
    # the frontal direction
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
