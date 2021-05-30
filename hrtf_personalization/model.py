import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EarMeasureEncoder(nn.Module):
    def __init__(self, ear_anthro_dim, ear_emb_dim):
        super(EarMeasureEncoder, self).__init__()
        self.ear_anthro_dim = ear_anthro_dim
        self.fc = nn.Sequential(
            nn.Linear(ear_anthro_dim, ear_emb_dim),
        )

    def forward(self, ear_anthro):
        assert ear_anthro.shape[1] == self.ear_anthro_dim
        return self.fc(ear_anthro)

class HeadMeasureEncoder(nn.Module):
    def __init__(self, head_anthro_dim, head_emb_dim):
        super(HeadMeasureEncoder, self).__init__()
        self.head_anthro_dim = head_anthro_dim
        self.fc = nn.Sequential(
            nn.Linear(head_anthro_dim, head_emb_dim),
        )

    def forward(self, head_anthro):
        assert head_anthro.shape[1] == self.head_anthro_dim
        return self.fc(head_anthro)


class ConvNNHrtfSht(nn.Module):
    def __init__(self, args):
        super(ConvNNHrtfSht, self).__init__()
        self.ear_enc = EarMeasureEncoder(args.ear_anthro_dim, args.ear_emb_dim)
        self.head_enc = HeadMeasureEncoder(args.head_anthro_dim, args.head_emb_dim)
        self.lr_enc = nn.Embedding(2, args.lr_emb_dim)
        self.freq_enc = nn.Embedding(args.freq_bin, args.freq_emb_dim)
        self.condition_dim = args.condition_dim
        emb_concat_dim = args.ear_emb_dim + args.head_emb_dim + args.freq_emb_dim
        emb_concat_dim += args.lr_emb_dim
        self.fc = nn.Linear(emb_concat_dim, args.condition_dim)
        self.norm = args.norm
        if self.norm == "batch":
            self.norm_method = nn.BatchNorm1d
        elif self.norm == "layer":
            self.norm_method = nn.GroupNorm
        elif self.norm == "instance":
            self.norm_method = nn.InstanceNorm1d
        else:
            raise ValueError("normalization method not recognized")

        self.conv1 = self.make_gen_block(1, 4, kernel_size=7, stride=3)
        self.conv2 = self.make_gen_block(4, 16, kernel_size=5, stride=2)
        self.conv3 = self.make_gen_block(16, 32, kernel_size=5, stride=2)
        self.conv4 = self.make_gen_block(32, 32, kernel_size=5, stride=3)
        if args.target == "hrtf":
            self.conv5 = self.make_gen_block(32, 440, kernel_size=5, stride=2, final_layer=True)
        elif args.target == "sht":
            self.conv5 = self.make_gen_block(32, 64, kernel_size=5, stride=2, final_layer=True)
        else:
            raise ValueError("training target not recognized")

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        if not final_layer:
            if self.norm == "layer":
                return nn.Sequential(
                    nn.Conv1d(input_channels, output_channels, kernel_size, stride),
                    self.norm_method(1, output_channels),
                    nn.ReLU()
                )
            else:
                return nn.Sequential(
                    nn.Conv1d(input_channels, output_channels, kernel_size, stride),
                    self.norm_method(output_channels),
                    nn.ReLU()
                )
        else:
            return nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size, stride),
            )

    def unsqueeze_condition(self, latent):
        return latent.view(len(latent), 1, self.condition_dim)

    def forward(self, ear_anthro, head_anthro, frequency, left_or_right):
        ear_anthro_encoding = self.ear_enc(ear_anthro)
        head_anthro_encoding = self.head_enc(head_anthro)
        frequency_encoding = self.freq_enc(frequency)
        left_or_right_enc = self.lr_enc(left_or_right)

        latent = torch.cat((ear_anthro_encoding, head_anthro_encoding, frequency_encoding), dim=1)
        latent = torch.cat((latent, left_or_right_enc), dim=1)
        latent = self.unsqueeze_condition(self.fc(latent))

        latent = self.conv1(latent)
        latent = self.conv2(latent)
        latent = self.conv3(latent)
        latent = self.conv4(latent)
        out = self.conv5(latent)

        return out
