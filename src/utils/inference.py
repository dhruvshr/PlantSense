"""
Model Inference Utils
"""
import torch
from PIL import Image
from src.utils.transforms import inference_transform
from src.datasets.plant_village import PlantVillage

def infer_image(image_path, model, device):
    # load image
    image = Image.open(image_path)
    image_tensor = inference_transform(image, device)
    base_dataset = PlantVillage()

    # perform inference
    with torch.no_grad():
        output = model(image_tensor)
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = base_dataset.plantvillage.classes[predicted_idx]
        # Convert confidence to percentage
        confidence_score = confidence.item() * 100

    return predicted_class, confidence_score