"""
Save model to pth file
"""

import os
import torch

MODEL_SAVE_DIR = 'models/'
CHECKPOINT_SAVE_DIR = MODEL_SAVE_DIR + 'checkpoints/'

def validate_location(location: str):
    """
    Validate model save location
    """
    if not os.path.exists(location):
        raise ValueError(f"Model save location {location} does not exist")
    
    return location

def save_model(model: torch.nn.Module, location: str, version: str):
    """
    Save model to pth file
    """
    validate_location(location)

    try:
        model_save_path = os.path.join(location, f'modelv{version}.pth')
        torch.save(model.state_dict(), model_save_path)
    except Exception as e:
        print(f"Error saving model: {e}")
        raise e

def save_model_checkpoints(
        model: torch.nn.Module,
        location: str,
        version: str,   
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        epoch: int,
        loss: float
    ):
    """
    Save model checkpoints to pth file
    """
    validate_location(location)

    checkpoint_dir = os.path.join(location, CHECKPOINT_SAVE_DIR)
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'criterion_state_dict': criterion.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'version': version
    }, os.path.join(checkpoint_dir, f'checkpointv{version}.pth'))
