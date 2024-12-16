"""
Atomic ResNet model
"""

import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, num_classes):
        """
        Initialize the ResNet model with a pretrained ResNet18 backbone.
        The model architecture consists of:
        1. A pretrained ResNet18 backbone with early layers frozen
        2. A modified fully connected layer with dropout for regularization
        3. Added batch normalization before average pooling for stability
        The model is designed for transfer learning with fine-tuning of later layers.
        
        Args:
            num_classes (int): Number of output classes
        """
        super(ResNet, self).__init__()
        
        # load pretrained resnet18 backbone
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        
        # freeze early layers
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
            
        # modify final layers for better feature extraction
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # add batch normalization for better training stability
        self.backbone.avgpool = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        """
        Forward pass through the ResNet model
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    