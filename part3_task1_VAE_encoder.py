import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------
# Device Configuration
# ---------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------------------------------
# Hyperparameters
# ---------------------------------------------------
BATCH_SIZE = 128
INPUT_SIZE = 28
HIDDEN_SIZE = 128
LATENT_DIM = 20

# ---------------------------------------------------
# MNIST Dataset
# ---------------------------------------------------
transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ---------------------------------------------------
# VAE Encoder with LSTM
# ---------------------------------------------------
class VAEEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, latent_dim):

        super(VAEEncoder, self).__init__()

        # Single-layer LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Mean vector
        self.fc_mu = nn.Linear(
            hidden_size,
            latent_dim
        )

        # Log variance vector
        self.fc_logvar = nn.Linear(
            hidden_size,
            latent_dim
        )

    # ---------------------------------------------------
    # Encoder Forward Pass
    # ---------------------------------------------------
    def encode(self, x):

        # x shape:
        # (batch_size, sequence_length, input_size)

        _, (hidden, _) = self.lstm(x)

        hidden = hidden[-1]

        mu = self.fc_mu(hidden)

        logvar = self.fc_logvar(hidden)

        return mu, logvar

    # ---------------------------------------------------
    # Reparameterization Trick
    # ---------------------------------------------------
    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)

        epsilon = torch.randn_like(std)

        z = mu + epsilon * std

        return z

    # ---------------------------------------------------
    # Full Forward Pass
    # ---------------------------------------------------
    def forward(self, x):

        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

# ---------------------------------------------------
# Initialize Encoder
# ---------------------------------------------------
model = VAEEncoder(
    INPUT_SIZE,
    HIDDEN_SIZE,
    LATENT_DIM
).to(device)

print("\nVAE Encoder Initialized Successfully.\n")

# ---------------------------------------------------
# Test Encoder Output
# ---------------------------------------------------
model.eval()

with torch.no_grad():

    images, labels = next(iter(test_loader))

    images = images.to(device)

    # Convert images into sequences
    # (batch, 1, 28, 28) -> (batch, 28, 28)
    sequences = images.squeeze(1)

    z, mu, logvar = model(sequences)

print("Input sequence shape:", sequences.shape)
print("Latent vector shape:", z.shape)
print("Mean vector shape:", mu.shape)
print("Log variance vector shape:", logvar.shape)

print("\nTask 1 completed successfully.")