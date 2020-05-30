import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class fcn_autoencoder(nn.Module):
    def __init__(self):
        super(fcn_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32 * 32 * 3, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 32 * 32 * 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z


class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
            # nn.Conv2d(48, 96, 4, stride=2, padding=1),         # [batch, 96, 2, 2]
            # nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            # nn.ReLU(),
		    nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
		    nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(32*32*3, 1024)
        self.fc21 = nn.Linear(1024, 64)
        self.fc22 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, 1024)
        self.fc4 = nn.Linear(1024, 32*32*3)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar, z

class VAE2d(nn.Module):
    def __init__(self, h_dim=2048, z_dim=64):
        super(VAE2d, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1), # 32 -> 16
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1), # 16 -> 8
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1), # 8 -> 4, total 256 * 4 * 4
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, stride=1, padding=1), # 4 -> 2, total 512 * 2 * 2
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Flatten()
        )
        self.decoder = nn.Sequential(
		    nn.ConvTranspose2d(512, 256, 3, stride=1),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 5, stride=1),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 9, stride=1),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 17, stride=1),
            nn.Sigmoid()
        )
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

    def encode(self, x):
        h = F.relu(self.encoder(x))
        return self.fc1(h), self.fc2(h)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = F.relu(self.fc3(z))
        z = z.view(z.size(0), 512, 2, 2)
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar, z


def loss_vae(recon_x, x, mu, logvar, criterion, is2d=False):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    if is2d:
        recon_x = recon_x.view(-1)
        x = x.view(-1)
    bce = criterion(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return bce + KLD