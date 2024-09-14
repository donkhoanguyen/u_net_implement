import os
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.amp
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Configuration
DEVICE = "cpu" #if torch.backends.mps.is_available() else "cpu"
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
CHECKPOINT_PATH = 'my_checkpoint.pth.tar'
TEST_FOLDER = 'carvana-image-masking-challenge/test'
OUTPUT_FOLDER = 'prediction'

# Define the model architecture
model = UNET(in_channels=3, out_channels=1).to(DEVICE)

# Load the trained model
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()  # Set the model to evaluation mode

load_checkpoint(CHECKPOINT_PATH)

# Define image transformation for test data
def load_image(image_path):
    # Load image as a PIL object
    image = Image.open(image_path).convert('RGB')
    # Convert PIL image to a NumPy array
    image = np.array(image)
    # Define transformations
    transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    # Apply transformations
    image = transform(image=image)["image"]
    return image.unsqueeze(0)  # Add batch dimension

# Predict mask for a single image
def predict_image(model, image_path):
    image = load_image(image_path).to(DEVICE)
    with torch.no_grad():
        output = model(image)
    return output

# Post-process the model output
def postprocess(output):
    output = torch.sigmoid(output)  # Apply sigmoid if you used it
    output = output.squeeze().cpu().numpy()  # Remove batch and channel dimensions
    output = (output > 0.5).astype(np.uint8)  # Binarize if necessary
    return output

# Save predictions for all images in the test folder
def save_predictions_as_imgs(test_folder, model, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(test_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(test_folder, filename)
            output = predict_image(model, img_path)
            processed_mask = postprocess(output)
            output_image = Image.fromarray(processed_mask * 255)  # Convert binary mask to image
            output_image.save(os.path.join(output_folder, filename.replace('.jpg', '_mask.png')))

# Run the prediction and save results
save_predictions_as_imgs(TEST_FOLDER, model, OUTPUT_FOLDER)
