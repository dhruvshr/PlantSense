"""
Atomic ResNet model
"""

import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        
        # Load pretrained ResNet18 backbone
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
            
        # Modify final layers for better feature extraction
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Add batch normalization for better training stability
        self.backbone.avgpool = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        return self.backbone(x)
    