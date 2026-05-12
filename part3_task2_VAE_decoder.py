import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

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
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

# ---------------------------------------------------
# MNIST Dataset
# ---------------------------------------------------
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ---------------------------------------------------
# Full VAE Model
# ---------------------------------------------------
class VAE(nn.Module):

    def __init__(self, input_size, hidden_size, latent_dim):

        super(VAE, self).__init__()

        # ---------------------------------------------------
        # Encoder
        # ---------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc_mu = nn.Linear(
            hidden_size,
            latent_dim
        )

        self.fc_logvar = nn.Linear(
            hidden_size,
            latent_dim
        )

        # ---------------------------------------------------
        # Decoder
        # ---------------------------------------------------
        self.decoder = nn.Sequential(

            nn.Linear(latent_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    # ---------------------------------------------------
    # Encoder
    # ---------------------------------------------------
    def encode(self, x):

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
    # Decoder
    # ---------------------------------------------------
    def decode(self, z):

        reconstruction = self.decoder(z)

        reconstruction = reconstruction.view(-1, 1, 28, 28)

        return reconstruction

    # ---------------------------------------------------
    # Forward Pass
    # ---------------------------------------------------
    def forward(self, x):

        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        reconstruction = self.decode(z)

        return reconstruction, mu, logvar

# ---------------------------------------------------
# Initialize Model
# ---------------------------------------------------
model = VAE(
    INPUT_SIZE,
    HIDDEN_SIZE,
    LATENT_DIM
).to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ---------------------------------------------------
# VAE Loss Function
# ---------------------------------------------------
def vae_loss(reconstruction, original, mu, logvar):

    # Reconstruction Loss
    reconstruction_loss = nn.functional.binary_cross_entropy(
        reconstruction,
        original,
        reduction='sum'
    )

    # KL Divergence
    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return reconstruction_loss + kl_loss

# ---------------------------------------------------
# Training Loop
# ---------------------------------------------------
print("\nTraining VAE...\n")

train_losses = []

for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0

    for images, _ in train_loader:

        images = images.to(device)

        # Convert image to sequence
        sequences = images.squeeze(1)

        optimizer.zero_grad()

        reconstruction, mu, logvar = model(sequences)

        loss = vae_loss(
            reconstruction,
            images,
            mu,
            logvar
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader.dataset)

    train_losses.append(average_loss)

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
        f"Loss: {average_loss:.4f}"
    )

print("\nTraining completed.")

# ---------------------------------------------------
# Reconstruction Visualization
# ---------------------------------------------------
model.eval()

with torch.no_grad():

    images, _ = next(iter(test_loader))

    images = images.to(device)

    sequences = images.squeeze(1)

    reconstruction, _, _ = model(sequences)

    images = images.cpu()

    reconstruction = reconstruction.cpu()

# ---------------------------------------------------
# Save Trained Model
# ---------------------------------------------------
torch.save(
    model.state_dict(),
    "vae_model.pth"
)

print("\nModel saved successfully.")

# ---------------------------------------------------
# TASK 3
# Random Latent Sampling + Generation
# ---------------------------------------------------
model.eval()

with torch.no_grad():

    # Sample from standard normal distribution
    random_z = torch.randn(
        10,
        LATENT_DIM
    ).to(device)

    # Generate images
    generated_images = model.decode(random_z)

    generated_images = generated_images.cpu()

print("\nGenerating plots...")

# ===================================================
# PLOT 1 — TRAINING LOSS
# ===================================================
plt.figure(figsize=(8,5))

plt.plot(train_losses)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VAE Training Loss")

plt.grid(True)

# ===================================================
# PLOT 2 — RECONSTRUCTION RESULTS
# ===================================================
plt.figure(figsize=(10,4))

n = 5

for i in range(n):

    # Original
    ax = plt.subplot(2, n, i + 1)

    plt.imshow(
        images[i].squeeze(),
        cmap='gray'
    )

    plt.title("Original")

    plt.axis('off')

    # Reconstruction
    ax = plt.subplot(2, n, i + 1 + n)

    plt.imshow(
        reconstruction[i].squeeze(),
        cmap='gray'
    )

    plt.title("Reconstructed")

    plt.axis('off')

plt.tight_layout()

# ===================================================
# PLOT 3 — GENERATED SAMPLES
# ===================================================
plt.figure(figsize=(12,2))

for i in range(10):

    ax = plt.subplot(1, 10, i + 1)

    plt.imshow(
        generated_images[i].squeeze(),
        cmap='gray'
    )

    plt.axis('off')

plt.suptitle("Generated MNIST Samples")

plt.tight_layout()

# ---------------------------------------------------
# Show All Plots At End
# ---------------------------------------------------
plt.show()

print("\nTask 2 and Task 3 completed successfully.")