import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ( 
    confusion_matrix, 
    precision_recall_fscore_support
)

from tqdm import tqdm

from src.utils.device import get_device
from src.datasets.plant_village import PlantVillage
from src.model.plantsense_resnet import PlantSenseResNetBase

class ModelEvaluator:
    def __init__(self, model, device, test_loader, class_names):
        """
        Comprehensive model evaluation utility
        
        Args:
            model (nn.Module): Trained PyTorch model
            device (torch.device): Inference device
            test_loader (DataLoader): Test data loader
            class_names (list): List of class names
        """
        self.model = model
        self.device = device
        self.test_loader = test_loader
        self.class_names = class_names

        self.all_labels = []
        self.all_preds = []
    
    def compute_metrics(self):
        """
        Compute comprehensive model performance metrics
        
        Returns:
            dict: Detailed performance metrics
        """
        self.model.eval()

        with torch.no_grad():
            test_loss = 0
            correct = 0
            total = 0
            
            with tqdm(self.test_loader, unit='batch', desc="Evaluating") as tepoch:
                for inputs, labels in tepoch:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                    
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    self.all_preds.extend(predicted.cpu().numpy())
                    self.all_labels.extend(labels.cpu().numpy())

                    tepoch.set_postfix(loss=test_loss / total, accuracy=100 * correct / total)
        
        # compute detailed metrics
        accuracy = 100 * correct / total
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.all_labels, 
            self.all_preds, 
            average='weighted',
            zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'loss': test_loss / len(self.test_loader),
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100
        }
    
    def plot_confusion_matrix(self, normalize=True):
        """
        Generate and plot confusion matrix
        
        Args:
            normalize (bool): Whether to normalize confusion matrix
        
        Returns:
            matplotlib.figure.Figure: Confusion matrix visualization
        """
        # Compute confusion matrix
        cm = confusion_matrix(
            self.all_labels, 
            self.all_preds
        )
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            cmap='Blues', 
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        
        return plt.gcf()
    
    def top_k_accuracy(self, k=3):
        """
        Compute top-k accuracy
        
        Args:
            k (int): Number of top predictions to consider
        
        Returns:
            float: Top-k accuracy percentage
        """
        self.model.eval()
        top_k_correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                _, top_k_preds = outputs.topk(k, dim=1)
                
                top_k_correct += torch.sum(
                    top_k_preds.eq(labels.view(-1, 1)).any(dim=1)
                ).item()
                total += labels.size(0)
        
        return 100 * top_k_correct / total
    
    def evaluate_and_explain(self, image):
        """
        Evaluate the model and provide explanations for the prediction
        """
        pass