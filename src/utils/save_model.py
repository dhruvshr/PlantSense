"""
Save model to pth file
"""

import os
import torch
from torchvision import models

MODEL_SAVE_DIR = 'models/'
CHECKPOINT_SAVE_DIR = MODEL_SAVE_DIR + 'checkpoints/'

def validate_directory(save_dir):
    """Checks if the directory exists. If not, creates it."""
    if not os.path.exists(save_dir):
        print(f"Directory {save_dir} does not exist. Creating it now...")
        os.makedirs(save_dir)
    else:
        print(f"Directory {save_dir} exists.")

def prompt_save_model():
    """Prompts user to choose how to save the model."""
    print("How would you like to save the model?")
    print("1. Save the entire model.")
    print("2. Save as a checkpoint (model state dict + optimizer).")
    print("3. Save with the default configuration.")
    choice = input("Please enter 1, 2, or 3: ")
    
    return choice.strip()

def save_model(model, save_dir, model_name, version):
    """Saves the entire model to the specified directory with versioning."""
    save_path = os.path.join(save_dir, f"{model_name}_v{version}_full_model.pth")
    torch.save(model, save_path)
    print(f"Model saved successfully at {save_path}")

def save_checkpoint(model, optimizer, save_dir, model_name, version, epoch=None):
    """Saves the model checkpoint (model state_dict and optimizer state_dict) with versioning."""
    save_path = os.path.join(f"{save_dir}", f"{model_name}_v{version}_checkpoint.pth")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch if epoch is not None else 0
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved successfully at {save_path}")

def save_with_default_config(model, save_dir, model_name, version):
    """Saves the model with the default configuration and versioning."""
    save_path = os.path.join(save_dir, f"{model_name}_v{version}_default_config.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
    }, save_path)
    print(f"Model saved with default configuration at {save_path}")

def save_trained_model(model, optimizer, save_dir, model_name, version, epoch=None):
    """Main function to interactively save a model with versioning."""
    # Validate the directory
    validate_directory(save_dir)
    
    # Prompt user for save option
    choice = prompt_save_model()
    
    # Save based on user choice
    if choice == '1':
        save_model(model, save_dir, model_name, version)
    elif choice == '2':

        save_checkpoint(model, optimizer, save_dir, model_name, version, epoch)
    elif choice == '3':
        save_with_default_config(model, save_dir, model_name, version)
    elif choice == '-1':
        return
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")

"""
if __name__ == '__main__':

    model = models.convnext_base(weights='DEFAULT')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    save_trained_model(
        model,
        optimizer,
        'model_files/',
        'test_model',
        1
    )
"""