import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.ToTensor()

train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

class FCAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32)  # latent space
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()  # because pixels are [0,1]
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        z = self.encoder(x)
        x_hat = self.decoder(z)
        x_hat = x_hat.view(x.size(0), 1, 28, 28)
        return x_hat
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

model = FCAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
test_losses = []

epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for x, _ in train_loader:
        x = x.to(device)
        
        x_hat = model(x)
        loss = criterion(x_hat, x)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_train_loss = total_loss/len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    test_loss = 0

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            x_hat = model(x)
            loss = criterion(x_hat, x)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    print(f"Epoch {epoch+1}, Loss: {avg_train_loss:.4f}, Test Loss:{avg_test_loss:.4f}")

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training vs Test Loss")
plt.show()

model.eval()

examples = next(iter(test_loader))
images, _ = examples
images = images.to(device)

with torch.no_grad():
    reconstructions = model(images)

n = 15

plt.figure(figsize=(10,4))

images = images.cpu()
reconstructions = reconstructions.cpu()

for i in range(n):
    # Original
    plt.subplot(2, n, i+1)
    plt.imshow(images[i].squeeze(), cmap='gray')
    plt.title(i+1)
    plt.axis('off')
    
    # Reconstructed
    plt.subplot(2, n, i+1+n)
    plt.imshow(reconstructions[i].squeeze(), cmap='gray')
    plt.title(i+1)
    plt.axis('off')

plt.show()

latent_vectors = []
labels = []

model.eval()

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        x_flat = x.view(x.size(0), -1)
        z = model.encoder(x_flat)
        
        latent_vectors.append(z.cpu())
        labels.append(y)

latent_vectors = torch.cat(latent_vectors)
labels = torch.cat(labels)

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
z_2d = tsne.fit_transform(latent_vectors.numpy())

plt.figure()
plt.scatter(z_2d[:,0], z_2d[:,1], c=labels, cmap='tab10', s=5)
plt.colorbar()
plt.title("t-SNE of Latent Space")
plt.show()