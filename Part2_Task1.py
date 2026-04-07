import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# -----------------------------
# 1. Data loading
# -----------------------------
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

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

# -----------------------------
# 2. Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -----------------------------
# 3. Model
# -----------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=28, hidden_size=64, num_layers=1, num_classes=10):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,   # single-layer LSTM
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: [batch_size, 28, 28]
        out, (hn, cn) = self.lstm(x)

        # final hidden representation
        embedding = out[:, -1, :]   # [B, hidden_size]

        logits = self.fc(embedding) # [B, 10]
        return logits, embedding

model = LSTMClassifier().to(device)

# -----------------------------
# 4. Loss and optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 5. Training settings
# -----------------------------
epochs = 10

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# -----------------------------
# 6. Training loop
# -----------------------------
for epoch in range(epochs):
    model.train()
    total_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        # [B, 1, 28, 28] -> [B, 28, 28]
        x = x.squeeze(1)

        logits, _ = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct_train += (preds == y).sum().item()
        total_train += y.size(0)

    avg_train_loss = total_train_loss / len(train_loader)
    train_acc = correct_train / total_train

    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)

    # -----------------------------
    # 7. Evaluation loop
    # -----------------------------
    model.eval()
    total_test_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            x = x.squeeze(1)

            logits, _ = model(x)
            loss = criterion(logits, y)

            total_test_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct_test += (preds == y).sum().item()
            total_test += y.size(0)

    avg_test_loss = total_test_loss / len(test_loader)
    test_acc = correct_test / total_test

    test_losses.append(avg_test_loss)
    test_accuracies.append(test_acc)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Test Loss: {avg_test_loss:.4f} | "
        f"Test Acc: {test_acc:.4f}"
    )

# -----------------------------
# 8. Final results
# -----------------------------
print("\nFinal Results")
print(f"Final Train Accuracy: {train_accuracies[-1] * 100:.2f}%")
print(f"Final Test Accuracy: {test_accuracies[-1] * 100:.2f}%")

# -----------------------------
# 9. Plot loss curves
# -----------------------------
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training vs Test Loss")
plt.legend()
plt.show()

# -----------------------------
# 10. Plot accuracy curves
# -----------------------------
plt.figure(figsize=(8, 4))
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Test Accuracy")
plt.legend()
plt.show()