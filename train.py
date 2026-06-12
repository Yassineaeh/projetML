import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torchvision import models

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
DATA_DIR = "dataset"

transform_train = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

transform_test = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

full_dataset = ImageFolder(DATA_DIR, transform=transform_train)

n = len(full_dataset)
n_train = int(0.7 * n)
n_val   = int(0.15 * n)
n_test  = n - n_train - n_val

train_set, val_set, test_set = random_split(full_dataset, [n_train, n_val, n_test])

train_dl = DataLoader(train_set, batch_size=32, shuffle=True)
val_dl   = DataLoader(val_set,   batch_size=32)
test_dl  = DataLoader(test_set,  batch_size=32)

print(f"Total : {n} | Train : {n_train} | Val : {n_val} | Test : {n_test}")




device = "cuda" if torch.cuda.is_available() else "cpu"
print("device :", device)

model = models.resnet18(weights="IMAGENET1K_V1")

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(512, 6)
model = model.to(device)

print("Modèle prêt")


optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    # Train
    model.train()
    total_loss = 0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()

    val_acc = correct / n_val
    print(f"Epoch {epoch+1}/10 — loss: {total_loss/len(train_dl):.3f} | val_acc: {val_acc:.3f}")


# Evaluation sur le test
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_dl:
        xb = xb.to(device)
        preds = model(xb).argmax(dim=1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(yb.numpy())

acc_test = (np.array(all_preds) == np.array(all_labels)).mean()
print(f"Accuracy test : {acc_test:.3f}")

# Matrice de confusion
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, colorbar=False, xticks_rotation=45)
plt.title("Matrice de confusion — Tri des déchets")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()


torch.save(model.state_dict(), "model.pth")
print("Modèle sauvegardé.")