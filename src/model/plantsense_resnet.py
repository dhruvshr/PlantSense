"""
PlantSense ResNet Model (base)
"""

import torch.nn as nn
import torchvision.models as models

class PlantSenseResNetBase(nn.Module):

    def __init__(self, num_classes, weights=None):
        """
            ResNet-18 based model for plant disease classification
            
            Args:
                num_classes (int): Number of disease classes
                pretrained (bool): Use ImageNet pretrained weights
            """
        super(PlantSenseResNetBase, self).__init__()

        # load base resnet model with default weights
        self.base_model = models.resnet18()
        # self.base_model = models.resnet18(
        #     weights=models.ResNet18_Weights.DEFAULT
        # )

        # # freeze base layers
        # for param in self.base_model.parameters():
        #     param.requires_grad = False

        # replace final fc layers
        num_features = self.base_model.fc.in_features
        # self.base_model.fc = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(num_features, num_classes)
        # )
        self.base_model.fc = nn.Linear(
            self.base_model.fc.in_features, 
            num_classes
        )

    def forward(self, x):
        return self.base_model(x)
    
    def unfreeze_layers(self, num_layers=10):
        """
        Progressively unfreeze model layers for fine-tuning

        Args:
            num_layers (int): Number of layers from the end to unfreeze
        """
        layers = list(self.base_model.parameters())[:-num_layers]
        for param in layers:
            param.requires_grad = True