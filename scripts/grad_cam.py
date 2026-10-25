import os
import cv2
import torch
import glob
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pickle
import matplotlib.pyplot as plt

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

model_path = "cnn_model.pth"
pickle_file = "preprocessed_data.pkl"
output_path = "gradcam_results"
output_figure_path = "gradcam_grid.png"
os.makedirs(output_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

model = CNNModel()
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

target_layers = [model.conv3]

with open(pickle_file, "rb") as f:
    dataset_splits = pickle.load(f)

test_samples = dataset_splits["test"][:15]

cam = GradCAM(model=model, target_layers=target_layers)

for i, (img_path, label) in enumerate(test_samples):
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    pil_img = Image.open(img_path).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    rgb_img = np.array(pil_img.resize((64, 64))) / 255.0
    targets = None
    greyscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    greyscale_cam = greyscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, greyscale_cam, use_rgb=True)
    visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    save_filename = f"label_{label}_img_{base_name}_gradcam.png"
    cv2.imwrite(os.path.join(output_path, save_filename), visualization_bgr)
    print(f"[{i+1}/{len(test_samples)}] Saved Individual image to: {save_filename}")

print("All heatmaps generated successfully.")

print("Compiling heatmaps into a grid...")

cam_images = glob.glob(os.path.join(output_path, "*_gradcam.png"))
cam_images.sort()

rows = 3
cols = 5
num_images_shown = rows * cols
images_to_plot = cam_images[:num_images_shown]

fig, axes = plt.subplots(rows, cols, figsize=(15, 9))
axes = axes.flatten()

for i in range(num_images_shown):
    ax = axes[i]

    if i < len(images_to_plot):
        img_path = images_to_plot[i]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        filename = os.path.basename(img_path)

        if "label_1" in filename:
            title_text = "Cancer (1)"
            color = "red"
        else:
            title_text = "Normal (0)"
            color = "green"
        
        ax.imshow(img)
        ax.set_title(title_text, fontsize=10, color=color, fontweight='bold')
    else:
        ax.axis("off")

plt.tight_layout()
plt.savefig(output_figure_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Grid of heatmaps saved to {output_figure_path}")