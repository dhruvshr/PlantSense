"""
ML Model Train script
"""
import os
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from dotenv import load_dotenv

from src.utils.device import get_device
from src.utils.save_model import save_trained_model
from src.datasets.plant_village import PlantVillage
from src.training.trainer import ModelTrainer
from src.model.plantsense_resnet import PlantSenseResNetBase
from src.utils.save_model import save_checkpoint, save_model

load_dotenv()

# def
BATCH_SIZE = 32
NUM_WORKERS = 2
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
CHECKPOINTS_SAVE_DIR = 'saved_models/checkpoints'
MODEL_SAVE_DIR = 'saved_models'
VERSION = 1_2

def train():
    # device configuration
    device = get_device()

    # load dataset
    base_dataset = PlantVillage()

    # split dataset
    train_data, val_data, test_data = base_dataset.split_dataset()

    # create dataloaders
    train_loader = DataLoader(
        train_data,
        batch_size = BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_data,
        batch_size = BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_data,
        batch_size = BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # init model
    model = PlantSenseResNetBase(
        num_classes=base_dataset.NUM_CLASSES,
    ).to(device)

    # setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    # create trainer
    trainer = ModelTrainer(
        model = model,
        device = device,
        criterion = criterion,
        optimizer = optimizer
    )

    print(f"Training {model.__class__.__name__} on {device}\n")

    accuracies = list()

    # training loop
    num_epochs = NUM_EPOCHS
    for epoch in range(num_epochs):
        train_metrics = trainer.train_epoch(train_loader, epoch, num_epochs)
        val_metrics = trainer.validate(val_loader, epoch, num_epochs)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.2f}%")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.2f}%")

        checkpoint_save = input("Save Checkpoint?: y or n: ")

        if (checkpoint_save.strip() == 'y'):
            save_checkpoint(
                model, 
                optimizer, 
                CHECKPOINTS_SAVE_DIR, 
                model.__class__.__name__, 
                VERSION, 
                epoch=epoch
            )
            pass
        elif (checkpoint_save.strip() == 'n'):
            pass

    # save model
    model_save = input("Save Model?: y or n: ")

    if (model_save.strip() == 'y'):
        save_model(
            model, 
            MODEL_SAVE_DIR, 
            model.__class__.__name__, 
            VERSION
        )
        pass
    elif (model_save.strip() == 'n'):
        pass

if __name__ == '__main__':
    train()
    
    



