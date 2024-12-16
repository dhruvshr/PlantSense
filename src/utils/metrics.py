"""
Metrics for training and evaluation
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class MetricsTracker:
    def __init__(self):
        """Initialize metrics dictionary to store training history"""
        self.history = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': None,
            'test_acc': None
        }
        
    def update_metrics(self, phase, loss, accuracy, epoch=None):
        """
        Update metrics for the specified phase
        
        Args:
            phase (str): 'train', 'val', or 'test'
            loss (float): Loss value
            accuracy (float): Accuracy value 
            epoch (int, optional): Training epoch number
        """
        # convert inputs to float
        loss = float(loss)
        accuracy = float(accuracy)
        
        if phase == 'test':
            self.history['test_loss'] = loss
            self.history['test_acc'] = accuracy
        else:
            # only append epoch number for the first phase in each epoch (e.g., 'train')
            if phase == 'train' and epoch is not None:
                self.history['epochs'].append(epoch)
            
            self.history[f'{phase}_loss'].append(loss)
            self.history[f'{phase}_acc'].append(accuracy)
            
    def plot_metrics(self, save_path=None, show=False):
        """
        Plot training and validation metrics
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        if not self.history['epochs']:
            print("No metrics data available to plot!")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        epochs = self.history['epochs']
        
        # Plot loss
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        if self.history['test_loss'] is not None:
            ax1.axhline(y=self.history['test_loss'], color='g', linestyle='--', 
                       label='Test Loss')
        ax1.set_title('Loss vs. Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy')
        if self.history['test_acc'] is not None:
            ax2.axhline(y=self.history['test_acc'], color='g', linestyle='--',
                       label='Test Accuracy')
        ax2.set_title('Accuracy vs. Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()
        
    def save_metrics(self, save_path):
        """
        Save metrics history to a file
        
        Args:
            save_path (str): Path to save the metrics
        """
        with open(save_path, 'w') as f:
            json.dump(self.history, f)
            
    def load_metrics(self, load_path):
        """
        Load metrics history from a file
        
        Args:
            load_path (str): Path to load the metrics from
        """
        with open(load_path, 'r') as f:
            self.history = json.load(f)

    def get_metrics(self):
        return self.history
