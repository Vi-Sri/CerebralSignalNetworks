import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, features),
        )

    def forward(self, x):
        return x + self.block(x)
    
def custom_loss_fn(x, encoded, latent_features, decoded, criterion):

    # print(f"inputsize {x.size()} encoded: {encoded.size()} deocded: {decoded.size()} latent: {latent_features.size()} " )

    # reconstruction_loss = criterion(decoded, x)
    feature_matching_loss = criterion(encoded, latent_features)
    # return reconstruction_loss + feature_matching_loss
    return feature_matching_loss

class EEGAutoencoderFC(nn.Module):
    def __init__(self, channels=128, time_freq=480, latent_dim=384, num_residual_blocks=2):
        super(EEGAutoencoderFC, self).__init__()
        # Encoder

        self.encoder = nn.Sequential(
            nn.Linear(channels*time_freq, 1000),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            *(ResidualBlock(1000) for _ in range(num_residual_blocks)),
            nn.Linear(1000, latent_dim), 
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1000), 
            nn.ReLU(),
            nn.Dropout(p=0.5),
            *(ResidualBlock(1000) for _ in range(num_residual_blocks)),
            nn.Linear(1000, channels*time_freq),
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,decoded

class EEGAutoencoder(nn.Module):
    def __init__(self, latent_dim=2048):
        super(EEGAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16*60, latent_dim),  # Fully connected layer
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16*60),  # Fully connected layer
            nn.ReLU(),
            nn.Unflatten(1, (16, 60)),
            nn.ConvTranspose1d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x