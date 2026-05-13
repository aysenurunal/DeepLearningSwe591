import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

input_size = 28
hidden_size = 128
num_layers = 1
latent_dim = 32
batch_size = 128
num_epochs = 30
learning_rate = 0.001

transform = transforms.ToTensor()

train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

class Encoder(nn.Module):     
    def __init__(self, input_size, hidden_size, latent_dim, num_layers=1):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        # x shape: (batch_size, 1, 28, 28)

        x = x.squeeze(1)  # -> (batch_size, 28, 28)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Use final hidden state
        hidden = hn[-1]

        mu = self.fc_mu(hidden)
        log_var = self.fc_logvar(hidden)

        return mu, log_var

class Sampling(nn.Module):
    def __init__(self):
        super(Sampling, self).__init__()

    def forward(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.ReLU(),

            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def forward(self, z):
        x_hat = self.decoder(z)
        x_hat = x_hat.view(-1, 1, 28, 28)
        return x_hat
    
class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, num_layers=1):
        super(VAE, self).__init__()

        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=hidden_size,
            latent_dim=latent_dim,
            num_layers=num_layers
        )

        self.sampling = Sampling()

        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        mu, log_var = self.encoder(x)

        z = self.sampling(mu, log_var)

        x_hat = self.decoder(z)

        return x_hat, mu, log_var
    
# Binary Cross Entropy reconstruction loss
bce_loss = nn.BCELoss(reduction='sum')


def loss_function(x_hat, x, mu, log_var):
    reconstruction_loss = bce_loss(x_hat, x)

    kl_divergence = -0.5 * torch.sum(
        1 + log_var - mu.pow(2) - log_var.exp()
    )

    return reconstruction_loss + kl_divergence

model = VAE(
    input_size=input_size,
    hidden_size=hidden_size,
    latent_dim=latent_dim,
    num_layers=num_layers
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model.train()

for epoch in range(num_epochs):
    total_loss = 0

    for images, _ in train_loader:
        images = images.to(device)

        optimizer.zero_grad()

        reconstructed, mu, log_var = model(images)

        loss = loss_function(reconstructed, images, mu, log_var)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader.dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

model.eval()

with torch.no_grad():
    images, _ = next(iter(test_loader))
    images = images.to(device)

    reconstructed, _, _ = model(images)

    images = images.cpu()
    reconstructed = reconstructed.cpu()


fig, axes = plt.subplots(2, 10, figsize=(15, 4))

for i in range(10):
    axes[0, i].imshow(images[i].squeeze(), cmap='gray')
    axes[0, i].axis('off')

    axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
    axes[1, i].axis('off')

axes[0, 0].set_title("Original")
axes[1, 0].set_title("Reconstructed")

plt.show()

model.eval()

with torch.no_grad():
    random_z = torch.randn(16, latent_dim).to(device)

    generated_images = model.decoder(random_z)

    generated_images = generated_images.cpu()


fig, axes = plt.subplots(4, 4, figsize=(8, 8))

for i, ax in enumerate(axes.flatten()):
    ax.imshow(generated_images[i].squeeze(), cmap='gray')
    ax.axis('off')

plt.suptitle("Generated MNIST Samples")
plt.show()