import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


# =========================================
# 1. Reproducibility
# =========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# =========================================
# 2. Hyperparameters
# =========================================
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128
NUM_CLASSES = 10


# =========================================
# 3. Device
# =========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# =========================================
# 4. Data
# MNIST image shape: [B, 1, 28, 28]
# We will convert it to [B, 28, 28] for LSTM
# =========================================
transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


# =========================================
# 5. Model
# Single-layer LSTM classifier
# =========================================
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=28, hidden_size=128, num_layers=1, num_classes=10):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,   # single-layer LSTM
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: [B, 28, 28]
        out, (hn, cn) = self.lstm(x)

        # Final hidden representation of the sequence
        embedding = out[:, -1, :]   # [B, hidden_size]

        logits = self.fc(embedding) # [B, 10]
        return logits, embedding


model = LSTMClassifier(hidden_size=HIDDEN_SIZE, num_layers=1, num_classes=NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# =========================================
# 6. Train / Evaluate Functions
# =========================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # [B, 1, 28, 28] -> [B, 28, 28]
        x = x.squeeze(1)

        logits, _ = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / len(loader)
    acc = correct / total
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        x = x.squeeze(1)

        logits, _ = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / len(loader)
    acc = correct / total
    return avg_loss, acc


# =========================================
# 7. Training Loop
# =========================================
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

best_test_acc = 0.0

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    if test_acc > best_test_acc:
        best_test_acc = test_acc

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
    )

print("\nTraining finished.")
print(f"Best Test Accuracy: {best_test_acc * 100:.2f}%")
print(f"Final Train Accuracy: {train_accuracies[-1] * 100:.2f}%")
print(f"Final Test Accuracy: {test_accuracies[-1] * 100:.2f}%")


# =========================================
# 8. Plot Training Curves
# =========================================
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training vs Test Loss")
plt.legend()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Test Accuracy")
plt.legend()
plt.show()


# =========================================
# 9. Extract Test Embeddings
# Final hidden state = embedding
# =========================================
@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    all_embeddings = []
    all_labels = []

    for x, y in loader:
        x = x.to(device).squeeze(1)
        logits, embedding = model(x)

        all_embeddings.append(embedding.cpu())
        all_labels.append(y)

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    return all_embeddings, all_labels


embeddings, labels = extract_embeddings(model, test_loader, device)

print("Embeddings shape:", embeddings.shape)
print("Labels shape:", labels.shape)


# =========================================
# 10. t-SNE
# Reduce embeddings to 2D for visualization
# =========================================
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(embeddings)

print("2D embeddings shape:", embeddings_2d.shape)


# =========================================
# 11. K-Means Clustering (k=10)
# =========================================
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
cluster_ids = kmeans.fit_predict(embeddings_2d)
centroids = kmeans.cluster_centers_


# =========================================
# 12. Plot t-SNE points + K-Means centroids
# =========================================
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embeddings_2d[:, 0],
    embeddings_2d[:, 1],
    c=labels,
    cmap="tab10",
    s=8,
    alpha=0.7
)

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=250,
    linewidths=2
)

plt.colorbar(scatter)
plt.title("t-SNE of LSTM Embeddings with K-Means Centroids")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()