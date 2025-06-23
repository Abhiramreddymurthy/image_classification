import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from alexnet_model import OptimizedAlexNet
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
batch_size = 32
num_epochs = 15
learning_rate = 0.0001
num_classes = 17

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_data = torchvision.datasets.ImageFolder(root='dataset/', transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OptimizedAlexNet(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "alexnet_optimized.pth")
