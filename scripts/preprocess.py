"""
Preprocessing Image Script
"""

from torch import Tensor
from PIL import Image
from src.utils.transforms import inference_transform



def preprocess_image(image: Image.Image, device: str) -> Tensor:
    """
    Preprocess image for inference
    """
    return inference_transform(image, device)