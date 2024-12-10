import torch 
from torchvision import models


for model in models.list_models():
    print(model)