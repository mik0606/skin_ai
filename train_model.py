import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
from torch.utils.data import DataLoader

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load dataset from folders
train_data = datasets.ImageFolder("../data/train", transform=transform)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

# Get number of classes from folder names
num_classes = len(train_data.classes)

# Load model and update final layer
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(5):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save trained model
torch.save(model, "model.pth")
print("✅ Model trained and saved!")
