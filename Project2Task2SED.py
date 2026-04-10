import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim


input_size = 28
hidden_size = 128
num_layers = 1
num_classes = 10
batch_size = 128
num_epochs = 10
learning_rate = 0.001

transform = transforms.ToTensor()

train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, (hn, cn) = self.lstm(x, (h0, c0))

        embedding = out[:, -1, :]

        out = self.fc(embedding)
        return out, embedding
    
def evaluate(loader):
    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images = images.squeeze(1)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

model = LSTMModel(input_size, hidden_size, num_layers, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

losses = []
train_accs = []
test_accs = []

for epoch in range(num_epochs):
    epoch_loss = 0

    for images, labels in train_loader:
        images = images.squeeze(1)

        outputs, _ = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)

    train_accs.append(evaluate(train_loader))
    test_accs.append(evaluate(test_loader))

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

train_acc = evaluate(train_loader)
test_acc = evaluate(test_loader)

print(f"Train Accuracy: {train_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")

embeddings = []
labels_list = []

model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        images = images.squeeze(1)
        outputs, emb = model(images)

        embeddings.append(emb)
        labels_list.append(labels)

embeddings = torch.cat(embeddings).numpy()
labels_list = torch.cat(labels_list).numpy()

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
emb_2d = tsne.fit_transform(embeddings)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(emb_2d)
centroids = kmeans.cluster_centers_

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(emb_2d[:,0], emb_2d[:,1], c=labels_list, cmap='tab10', s=10)
plt.colorbar()
plt.scatter(centroids[:,0], centroids[:,1], c='black', s=200, marker='X')
plt.title("t-SNE of LSTM Embeddings")
plt.show()

plt.plot(losses)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.plot(train_accs, label="Train Accuracy")
plt.plot(test_accs, label="Test Accuracy")
plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy %")
plt.legend()
plt.show()