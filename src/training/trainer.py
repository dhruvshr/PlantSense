"""
Model Trainer
"""

import torch
from tqdm import tqdm

class ModelTrainer():
    def __init__(self, 
                 model, 
                 device, 
                 criterion, 
                 optimizer, 
                 scheduler=None):
        """
        Training and validation logic for ml model

        Args:
            model (nn.Module): PyTorch model
            device (torch.device): Training device (CPU/GPU)
            criterion (nn.Module): Loss function
            optimizer (torch.optim): Optimizer
            scheduler (optional): Learning rate scheduler
        """
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_epoch(self, train_loader, epoch: int, num_epochs: int):
        """
        Train model for one epoch
        
        Args:
            train_loader (DataLoader): Training data loader
        
        Returns:
            dict: Epoch metrics (loss, accuracy)
        """
        self.model.train()
        running_loss = 0
        correct = 0
        total = 0

        with tqdm(
            train_loader,
            unit='batch',
            desc=f"Training Epoch {epoch+1}/{num_epochs}",
        ) as tepoch:
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                # forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # backward pass
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # update progress bar with loss and accuracy
                tepoch.set_postfix(
                    loss = running_loss / (total + 1),
                    accuracy = 100 * correct / total   
                )
        
        return {
            'loss': running_loss / len(train_loader),
            'accuracy': 100 * correct / total
        }
    
    def validate(self, val_loader, epoch: int, num_epochs: int):
        """
        Validate model performance
        
        Args:
            val_loader (DataLoader): Validation data loader
        
        Returns:
            dict: Validation metrics
        """
        self.model.eval()
        valid_loss = 0
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():
            with tqdm(
                val_loader, 
                unit = 'batch',
                desc = f"Validating Epoch {epoch+1}/{num_epochs}",
            ) as vepoch:
                for inputs, labels in vepoch:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    valid_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    valid_correct += (predicted == labels).sum().item()
                    valid_total += labels.size(0)

                    vepoch.set_postfix(
                        loss = valid_loss / (valid_total + 1),
                        accuracy = 100 * valid_correct / valid_total  
                    )
        return {
            'loss': valid_loss / len(val_loader),
            'accuracy': 100 * valid_correct / valid_total
        }
