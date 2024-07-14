# Imports
import numpy as np # linear algebra

# Imports Torch
import torch
import torch.nn as nn
import pytorch_lightning as pl
from audio_encoders_pytorch import AutoEncoder1d

from audio_encoders_pytorch import AutoEncoder1d #from model import AutoEncoder1d

import auraloss
import torch.nn as nn

# I improve audio quality (from simple MSE used before)
# By using this Multiresolution Time Frequency loss
# It reduces the high frequency flicering I noticed using MSE Loss (time domain loss)

class TimeFrequencyLoss(nn.Module):
    def __init__(self, alpha, tau, gain, sr, duration):
        super().__init__()
        
        # Sample size ~6000
        self.frequ_loss = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[32, 128, 512, 2048], #[32, 128, 512, 2048, 8192, 32768]
            hop_sizes=[16, 64, 256, 1024], #[16, 64, 256, 1024, 4096, 16384]
            win_lengths=[32, 128, 512, 2048], #[32, 128, 512, 2048, 8192, 32768]
            w_sc=0.0,
            w_phs=0.0,
            w_lin_mag=1.0,
            w_log_mag=1.0,
        )
        self.time_loss = nn.MSELoss()
        self.alpha = alpha
        
        length = int(sr * duration)
        t = torch.linspace(0,1,length)
        self.enveloppe = 1 + gain * torch.exp(-t/tau)
        

    def forward(self, y_hat, y):
        enveloppe = self.enveloppe.to(y_hat.device)
        
        y_hat_mod = y_hat * enveloppe
        y_mod = y * enveloppe
        
        
        # Calculate frequency domain loss
        f_loss = self.frequ_loss(y_hat, y)
        
        # Calculate time domain loss
        t_loss = self.time_loss(y_hat, y)
        
        # Combine the losses
        total_loss = f_loss + self.alpha * t_loss
        return total_loss

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, model_cfg, training_cfg, data_cfg):
        super(LitAutoEncoder, self).__init__()
        self.save_hyperparameters() # for wandb
        
        # High level features of the AutoEncoder
        embedded_length = int(data_cfg['duration'] * data_cfg['sr'])
        for factor in model_cfg['factors']:
            embedded_length = int((embedded_length-1)/factor)+1

        input_channels = model_cfg['channels']
        for multiplier in model_cfg['multipliers']:
            input_channels = input_channels // multiplier
        
        self.length = int(data_cfg['duration'] * data_cfg['sr'])
        self.compression_rate = model_cfg['compression_rate']
        
        self.conv_out_channels = model_cfg['channels']
        self.conv_out_length = embedded_length
        self.conv_out_dim = self.conv_out_channels * self.conv_out_length
        self.latent_dim = int(self.length * self.compression_rate)

        # Model initialization
        self.model = AutoEncoder1d(
            in_channels=model_cfg['in_channels'],
            channels=model_cfg['channels'],
            multipliers=model_cfg['multipliers'],
            factors=model_cfg['factors'],
            num_blocks=model_cfg['num_blocks']
        )
        
        self.encode_linear = nn.Linear(self.conv_out_dim, self.latent_dim)
        self.decode_linear = nn.Linear(self.latent_dim, self.conv_out_dim)
        self.features_linear = nn.Linear(self.latent_dim, 3)
        
        print(f'{"Global Compression Rate":>30} : {100 * self.compression_rate:.2f} % (~{int(0.5+1/self.compression_rate)})')
        print(f'{"Convolution Compression Rate":>30} : {100 * self.conv_out_dim / self.length:.2f} % (~{int(0.5+self.length/self.conv_out_dim)})')
        print(f'{"Dense Compression Rate":>30} : {100 * self.latent_dim / self.conv_out_dim:.2f} % (~{int(0.5+self.conv_out_dim/self.latent_dim)})')
        print(f'{"Input Shape":>30} : ({input_channels},{self.length})')
        print(f'{"Conv Latent Shape":>30} : ({self.conv_out_channels},{self.conv_out_length}) -> {self.conv_out_dim}')
        print(f'{"Latent Shape":>30} : ({self.latent_dim})')
        print(f'{"Parameter Number":>30} : ({self.count_parameters()})') 
        print(f'{"Encoder + Decoder are":>30} : Resnet-{int(np.sum(model_cfg["num_blocks"])):_}')
        
        # Training 
        self.lr = training_cfg['lr']
        self.sr = data_cfg['sr']
        
        
        self.audio_loss = TimeFrequencyLoss(**training_cfg['audio_loss_params'],
                                         sr = training_cfg['sr'],
                                         duration = training_cfg['duration'],
                                        )
        self.beta = training_cfg['features_loss_params']['beta']
        self.features_loss = nn.MSELoss()
        

        # Placeholder for the first batch of audio samples
        self.first_audio_sample = None
        
    def forward(self, x):
        z = self.forward_encode(x)
        f = self.forward_features(z)
        x = self.forward_decode(z)
        return x, z, f
    
    def forward_features(self, z):
        f = self.features_linear(z)
        return f
    
    def forward_encode(self, x):
        z = self.model.encode(x)
        z = z.flatten(1)
        z = self.encode_linear(z)
        return z
    
    def forward_decode(self,z):
        z = self.decode_linear(z)
        z = z.view(z.shape[0],self.conv_out_channels,self.conv_out_length)
        x = self.model.decode(z)
        return x[..., :self.length]
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

