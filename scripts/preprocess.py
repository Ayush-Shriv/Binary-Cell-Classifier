import os
import glob
import random
import pickle
from PIL import Image
import numpy as np

normal_folder = "data/normal"
cancer_folder = "data/cancer"

normal_images = glob.glob(os.path.join(normal_folder, "*.tif"))
cancer_images = glob.glob(os.path.join(cancer_folder, "*.tif"))

normal_data = [(img_path, 0) for img_path in normal_images]
cancer_data = [(img_path, 1) for img_path in cancer_images]

combined_data = normal_data + cancer_data
random.shuffle(combined_data)

total_size = len(combined_data)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)

test_size = total_size - train_size - val_size
train_data = combined_data[:train_size]
val_data = combined_data[train_size:train_size + val_size]
test_data = combined_data[train_size + val_size:]

def save_data(train_data, val_data, test_data, filename="preprocessed_data.pkl"):
    combined_data = {
            "train": train_data,
            "val": val_data,
            "test": test_data
        }
    with open(filename, "wb") as f:
        pickle.dump(combined_data, f, pickle.HIGHEST_PROTOCOL)
    
    print(f"Saved to {filename}")

save_data(train_data, val_data, test_data)