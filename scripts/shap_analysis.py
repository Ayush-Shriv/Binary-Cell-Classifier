import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import pickle

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

class PreSigmoidWrapper(nn.Module):
    def __init__(self, model):
        super(PreSigmoidWrapper, self).__init__()
        self.original_model = model

    def forward(self, x):
        x = self.original_model.pool1(F.relu(self.original_model.conv1(x)))
        x = self.original_model.pool2(F.relu(self.original_model.conv2(x)))
        x = self.original_model.pool3(F.relu(self.original_model.conv3(x)))
        x = self.original_model.flatten(x)
        x = F.relu(self.original_model.fc1(x))
        x = self.original_model.fc2(x)
        return x

model_path = "cnn_model.pth"
pickle_file = "preprocessed_data.pkl"
output_path = "shap_results"
os.makedirs(output_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

model = CNNModel()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

explainer_model = PreSigmoidWrapper(model)
explainer_model.to(device)
explainer_model.eval()

with open(pickle_file, "rb") as f:
    dataset_splits = pickle.load(f)

background_samples = dataset_splits["train"][::5][:50]
background_tensors = torch.stack([
    transform(Image.open(path).convert("RGB")) for path, _ in background_samples]).to(device)

test_samples = dataset_splits["test"][:50:2]
test_tensors = torch.stack([
    transform(Image.open(path).convert("RGB")) for path, _ in test_samples]).to(device)

explainer = shap.GradientExplainer(explainer_model, background_tensors)
shap_values = explainer.shap_values(test_tensors)

if isinstance(shap_values, list):
    class_shap_values = shap_values[0]
else:
    class_shap_values = shap_values

print("Type:", type(shap_values))
print("Shape:", np.array(class_shap_values).shape)
print("Dimensions:", np.array(class_shap_values).ndim)

test_images_nhwc = test_tensors.cpu().numpy().transpose(0, 2, 3, 1)
class_shap_values = np.array(class_shap_values)
print("Raw SHAP shape:", class_shap_values.shape)
if class_shap_values.ndim == 5:
    class_shap_values = np.squeeze(class_shap_values, axis=-1)
shap_values_nhwc = np.transpose(class_shap_values, (0, 2, 3, 1))
print("Final SHAP shape:", shap_values_nhwc.shape)

print("Saving individual SHAP heatmaps...")
for i, (img_path, _) in enumerate(test_samples):
    plt.figure()
    shap.image_plot([shap_values_nhwc[i:i+1]],
                    test_images_nhwc[i:i+1],
                    show=False)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    plt.savefig(os.path.join(output_path, f"{base_name}_shap.png"), bbox_inches='tight')
    plt.close()

avg_shap_values = np.mean(np.abs(shap_values_nhwc), axis=(0, -1))

plt.figure(figsize=(6, 6))
plt.imshow(avg_shap_values, cmap='jet')
plt.title("Average Absolute SHAP Importance Map")
plt.colorbar(label="Mean Absolute SHAP Value")
plt.axis('off')
plt.savefig(os.path.join(output_path, "average_absolute_shap_map.png"), bbox_inches='tight')
plt.close("all")

print("\n--- SHAP Analysis Summary ---")
print(f"Processed {len(test_samples)} test images against a background of {len(background_samples)} samples.")
print(f"All visualizations sucessfully saved to: '{output_path}/'")
print("SHAP Analysis completed.")