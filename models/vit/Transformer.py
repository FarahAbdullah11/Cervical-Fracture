import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# =========================
# PREPROCESSING
# =========================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1)
    ),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# LOAD DATASET
# =========================

train_dataset = datasets.ImageFolder(
    "../../cervical fracture/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    "../../cervical fracture/val",
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)

print("Classes:", train_dataset.classes)

# =========================
# LOAD ViT MODEL
# =========================

model = timm.create_model(
    'vit_tiny_patch16_224',
    pretrained=True,
    num_classes=2
)

model = model.to(device)

# =========================
# LOSS + OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4
)

# =========================
# TRAINING
# =========================

epochs = 10

for epoch in range(epochs):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    avg_loss = running_loss / len(train_loader)

    print(f"\nEpoch [{epoch+1}/{epochs}]")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")

# =========================
# VALIDATION
# =========================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

val_accuracy = 100 * correct / total

print(f"\nValidation Accuracy: {val_accuracy:.2f}%")

# =========================
# SAVE MODEL
# =========================

torch.save(model.state_dict(), "../../vit_model.pth")