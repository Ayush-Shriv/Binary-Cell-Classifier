import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from preprocess import save_data
import pickle
import os
import glob
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import torchvision.transforms as transforms
from sklearn.metrics import(
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    classification_report
)

class PickleDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool3 = nn.MaxPool2d((2, 2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))   
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
model = CNNModel()
model.load_state_dict(torch.load("cnn_model.pth"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

test_pkl_path = "preprocessed_data.pkl"
model_path = "cnn_model.pth"
BATCH_SIZE = 32
TRESHOLD = 0.5

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

if not os.path.exists(test_pkl_path):
    raise FileNotFoundError(f"Pickle file {test_pkl_path} not found.")

with open(test_pkl_path, "rb") as f:
    data = pickle.load(f)

test_dataset = PickleDataset(data["test"], transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Test Samples:", len(test_dataset))

model = CNNModel()

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found.")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("Loaded model weights from:", model_path)

all_probs = []
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = outputs.view(-1)
        preds = (probs >= TRESHOLD).int()

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print("\n===== Evaluation Metrics =====")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, zero_division=0))

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal (0)", "Cancer (1)"],
    yticklabels=["Normal (0)", "Cancer (1)"],
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
conf_matrix_path = "confusion_matrix.png"
plt.savefig(conf_matrix_path, dpi=300, bbox_inches="tight")
print(f"\nConfusion matrix saved to: {conf_matrix_path}")
plt.show()

fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = roc_auc_score(all_labels, all_probs)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

roc_curve_path = "roc_curve.png"
plt.tight_layout()
plt.savefig(roc_curve_path)

print(f"ROC curve saved to: {roc_curve_path}")
print(f"AUC Score: {roc_auc:.4f}")
plt.show()