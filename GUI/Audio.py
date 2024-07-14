import numpy as np
import time

# Imports
import numpy as np # linear algebra
import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Imports Torch
import torch
import torchaudio

# Imports Autre
import pickle
import yaml
from IPython.display import Audio
import soundfile as sf
import matplotlib.pyplot as plt


from deep_ae import LitAutoEncoder, TimeFrequencyLoss

class LatentPlayGenerator(LitAutoEncoder):

    def __init__(self, data_path, device = 'cpu'):

        path_cfg  = os.path.join(data_path,'config.yaml')
        path_ckpt = os.path.join(data_path,'best-checkpoint.ckpt')
        path_plk  = os.path.join(data_path,'latent_play_parameters.pkl')

        with open(path_cfg, 'r') as file:
            all_config = yaml.safe_load(file)
        #all_config = cfg

        
        super().__init__(
            all_config['MODEL'],
            all_config['TRAINING'],
            all_config['DATA'],
        )
        

        #self.load_from_checkpoint(path_ckpt)
        # Load the model checkpoint from the specified path
        #checkpoint = LitAutoEncoder.load_from_checkpoint(path_ckpt)
        checkpoint = torch.load(path_ckpt, map_location=torch.device(device))
        self.load_state_dict(checkpoint['state_dict'])

        # To load the dictionary of arrays from the file
        with open(path_plk, 'rb') as file:
            latent_play_parameters = pickle.load(file)

        self.A = np.stack([
            latent_play_parameters["z_pca_1"], 
            latent_play_parameters["z_pca_2"], 
            latent_play_parameters["theta_freq"], 
            latent_play_parameters["theta_attack"], 
            latent_play_parameters["theta_release"],
                           ])
        self.C = self.A.T @ np.linalg.inv(self.A @ self.A.T)
        self.z = None
        self.min_max_pca_1 = latent_play_parameters["z_pca_1_scale"]
        self.min_max_pca_2 = latent_play_parameters["z_pca_2_scale"]
        self.device_name = device

    
    def encode(self, file):
        # Load audio file
        waveform, sample_rate = torchaudio.load(file, normalize=True)

        # Ensure mono by averaging channels if necessary
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if sample rate is different from cfg
        if sample_rate != self.sr:
            transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sr)
            waveform = transform(waveform)

        # Normalize the waveform
        waveform = waveform / waveform.abs().max()

        # Pad or trim to the desired sample length
        if waveform.size(1) < self.length:
            pad_size = self.length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        elif waveform.size(1) > self.length:
            waveform = waveform[:, :self.length]
        
         # Apply linear fade-out to the last 10% of the audio
        fade_out_length = int(self.length * 0.1)
        fade_out = torch.linspace(1, 0, fade_out_length)

        # Apply fade out to the last 10% of the waveform
        waveform[:, -fade_out_length:] *= fade_out

        self.z = self.forward_encode(waveform.unsqueeze(0)).squeeze().detach().numpy().flatten()
        b = self.A @ self.z

        b[0], b[1] = self.norm_pca(b[0], b[1])
        return b[0], b[1], b[2], b[3], b[4]
    
    def decode(self,
            latent_pca1,
            latent_pca2,
            target_freq,
            target_attack,
            target_release,
            ):
        
        latent_pca1, latent_pca2 = self.denorm_pca(latent_pca1, latent_pca2)
        target_freq, target_attack, target_release = target_freq, target_attack, target_release
        # Make z satisfy the targets with minimum change according linear regression
        b = np.array([latent_pca1, latent_pca2, target_freq, target_attack, target_release])
        z_prim = self.z - self.C @ (self.A @ self.z - b)

        z_prim = torch.tensor(z_prim).unsqueeze(0).float()
        x_hat = self.forward_decode(z_prim).squeeze().detach().numpy()
        return x_hat
    
    def denorm_pca(self, latent_pca1, latent_pca2):
        latent_pca1 = 2 * ((latent_pca1 + 1)/2 ) * (self.min_max_pca_1[1] - self.min_max_pca_1[0]) + self.min_max_pca_1[0]
        latent_pca2 = 2 * ((latent_pca2 + 1)/2 ) * (self.min_max_pca_2[1] - self.min_max_pca_2[0]) + self.min_max_pca_2[0]
        return latent_pca1, latent_pca2

    def norm_pca(self, latent_pca1, latent_pca2):
        latent_pca1 = 2 * (latent_pca1 - self.min_max_pca_1[0]) / (self.min_max_pca_1[1] - self.min_max_pca_1[0]) - 1
        latent_pca2 = 2 * (latent_pca2 - self.min_max_pca_2[0]) / (self.min_max_pca_2[1] - self.min_max_pca_2[0]) - 1
        return latent_pca1, latent_pca2


if __name__ == '__main__':
    model = LatentPlayGenerator(r'models\RUN_8\data')
    latent_pca1, latent_pca2, target_freq, target_attack, target_release = model.encode('Dataset/kick_dataset/Acoustic Kicks/KSHMR Acoustic Kick 01.wav')
    x_hat = model.decode(latent_pca1, latent_pca2, target_freq, target_attack, target_release)

    # x hat shaped (T)
    # Plot the result
    plt.figure(figsize=(10, 4))
    plt.plot(x_hat)
    plt.title('Reconstructed Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()
