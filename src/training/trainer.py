"""
Model Trainer
"""

import torch
from tqdm import tqdm
from torch import nn as nn
from torch import optim
from src.utils.metrics import MetricsTracker

SAVE_MODEL_PATH = 'saved_models/checkpoints/resnet_model_best_2.pth'

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
        self.metrics = MetricsTracker(model=self.model)

    def train_epoch(self, train_loader, epoch: int, num_epochs: int):
        """
        Train model for one epoch
        
        Args:
            train_loader (DataLoader): Training data loader
        
        Returns:
            dict: Epoch metrics (loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                # gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': running_loss/total,
                    'acc': 100.*correct/total
                })
    
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


    def train_model(self, model, train_loader, val_loader, device, num_epochs=10):
        # Loss function with label smoothing for better generalization
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Optimizer with weight decay for regularization
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.001,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3
        )
        
        # Early stopping
        best_val_acc = 0
        patience = 5
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # train
            self.train_epoch(train_loader, epoch, num_epochs)
            
            # validate
            val_metrics = self.validate(val_loader, epoch, num_epochs)

            # Early stopping check
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), SAVE_MODEL_PATH)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping triggered after epoch {epoch+1}')
                    break

        return model