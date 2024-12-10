import torch
import cv2
import os

from torchvision import transforms

# load plant village dataset
DATA_DRIVE = os.getenv('DATA_DRIVE')
plant_village_path = DATA_DRIVE + 'Plant_leave_diseases_dataset_with_augmentation/'

# load dataset