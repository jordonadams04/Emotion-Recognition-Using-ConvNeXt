# ConvNeXt Emotion Recognition

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2
import time
from ptflops import get_model_complexity_info
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# Check GPU
print("Using device:", torch.cuda.get_device_name())

# Constants
BATCH_SIZE = 32
NUM_CLASSES = 7
IMG_SIZE = 224
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load preprocessed data
print("Loading preprocessed .npz data...")
data = np.load('/content/processed_data_128.npz')
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

# Split validation set from training
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# Dataset wrapper
class FERDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = np.argmax(labels, axis=1)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.uint8)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

# Transforms with lazy resizing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

train_loader = DataLoader(FERDataset(x_train, y_train, transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(FERDataset(x_val, y_val, transform), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(FERDataset(x_test, y_test, transform), batch_size=BATCH_SIZE, shuffle=False)

# Load pretrained ConvNeXt-Tiny
import timm
model = timm.create_model('convnext_tiny', pretrained=True, num_classes=NUM_CLASSES)
model = model.to(DEVICE)

# Compute class weights
flat_labels = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(NUM_CLASSES), y=flat_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# Training setup
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)  # Adjusted LR

# Full training loop with validation and best model saving
print("\n--- Starting Training ---")
best_val_acc = 0.0
for epoch in range(EPOCHS):
    total_loss = 0
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = 100 * correct / total
    print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt")
        print("Best model saved.")

# Load best model
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# Evaluation
print("\n--- Evaluating Model ---")
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# === Profiling ===
print("\n--- Profiling Model ---")

# 1. Parameter count
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Parameters: {params:,} (~{params/1e6:.2f}M)")

# 2. FLOPs (using ptflops)
with torch.cuda.device(0):
    macs, _ = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
    print(f"FLOPs: {macs}")

# 3. Inference Time Breakdown
rand_idx = np.random.randint(len(x_test))
sample_img = x_test[rand_idx]
sample_tensor = transform(Image.fromarray(sample_img.astype(np.uint8))).unsqueeze(0).to(DEVICE)

# Preprocessing Time
start = time.time()
preprocessed = transform(Image.fromarray(sample_img.astype(np.uint8))).unsqueeze(0).to(DEVICE)
pre_time = time.time() - start

# Forward Pass Time
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
starter.record()
with torch.no_grad():
    output = model(preprocessed)
ender.record()
torch.cuda.synchronize()
forward_time = starter.elapsed_time(ender) / 1000  # in seconds

# Postprocessing Time
start = time.time()
pred_label = torch.argmax(output, dim=1).item()
true_label = np.argmax(y_test[rand_idx])
post_time = time.time() - start

print(f"Preprocessing Time: {pre_time*1000:.2f} ms")
print(f"Forward Pass Time: {forward_time*1000:.2f} ms")
print(f"Postprocessing Time: {post_time*1000:.2f} ms")

# 4. Memory Usage
max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
print(f"Max GPU Memory Allocated: {max_mem:.2f} MB")

# 5. Visualize prediction
label_map = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
plt.imshow(sample_img)
plt.title(f"True: {label_map[true_label]} | Predicted: {label_map[pred_label]}")
plt.axis('off')
plt.show()
