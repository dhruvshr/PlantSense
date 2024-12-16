"""
ML Model Train script
"""

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from dotenv import load_dotenv

from src.utils.device import get_device
from src.datasets.plant_village import PlantVillage
from src.training.trainer import ModelTrainer
from src.model.base_resnet import ResNet
from src.utils.transforms import optimized_transform

load_dotenv()

# training parameters
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

def train():
    # Device configuration
    device = get_device()
    
    # Dataset with optimized transforms
    train_transform = optimized_transform(is_training=True)

    # Load dataset
    train_dataset = PlantVillage(transform=train_transform)
    train_data, val_data, test_data = train_dataset.split_dataset()
    
    # create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # init model
    model = ResNet(
        num_classes=train_dataset.NUM_CLASSES
    ).to(device)

    # setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr = LEARNING_RATE
    )

    # create trainer
    trainer = ModelTrainer(
        model = model,
        device = device,
        criterion = criterion,
        optimizer = optimizer
    )
    
    # train model
    trained_model = trainer.train_model(
        model,
        train_loader,
        val_loader,
        device
    )

    print("\nTraining Metrics:")
    # show metrics
    metrics = trainer.metrics.get_metrics()
    for epoch in range(len(metrics['epochs'])):
        print(f"Epoch {metrics['epochs'][epoch]}")
        print(f"Train Loss: {metrics['train_loss'][epoch]:.4f}", end=' | ')
        print(f"Train Accuracy: {metrics['train_acc'][epoch]:.2f}%", end=' | ')
        print(f"Val Loss: {metrics['val_loss'][epoch]:.4f}", end=' | ')
        print(f"Val Accuracy: {metrics['val_acc'][epoch]:.2f}%")

    # plot all metrics
    trainer.metrics.plot_metrics(save_path='results/metrics.png')

if __name__ == '__main__':
    train()
    
    



