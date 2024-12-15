"""
Model Loader Utils
"""
import torch
from src.model.plantsense_resnet import PlantSenseResNetBase
from src.datasets.plant_village import PlantVillage
from src.utils.device import get_device

MODEL_PATH = 'saved_models/modelv1_1.pth'

def load_model(device=None):
    if device is None:
        device = get_device()

    # load model (modify as per training script)
    model = PlantSenseResNetBase(
        num_classes=PlantVillage().NUM_CLASSES
    ).to(device)

    # load the state dict and modify the keys to match model structure
    state_dict = torch.load(MODEL_PATH, weights_only=True)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict[f"base_model.{key}"] = value
    
    # load the modified state dict
    model.load_state_dict(new_state_dict)
    model.eval()

    return model