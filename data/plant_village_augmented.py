from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

import cv2
import os
import dotenv

from data.dataset_interface import DatasetInterface

dotenv.load_dotenv()
DATA_DRIVE = os.getenv('DATA_DRIVE')

class PlantVillageAugmented(DatasetInterface):
    def __init__(self, 
                 name: str = 'PlantVillageAugmented', 
                 path: str = DATA_DRIVE + 'Plant_leave_diseases_dataset_with_augmentation/',
                 augmented: bool = True):
        super().__init__(name = name, path = path)
        self.augmented = augmented
        
        # loading the data
        self.load_data()

    def load_data(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """
        Load the PlantVillage dataset from disk.
        """
        dataset = datasets.ImageFolder(self.path)

        self.dataset = dataset
        self.data = dataset.imgs
        self.classes = dataset.classes
        self.targets = dataset.targets
        self.class_idx = dataset.class_to_idx
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

    def get_classes_idx(self) -> dict[str, int]:
        """
        Map class labels to indices.
        """
        return self.class_idx
    
    def get_classes(self) -> list[str]:
        """
        Get the list of class labels in the dataset.
        """
        return self.classes
    
    def head(self, num_samples: int = 5):
        """
        Prints the first few data samples and labels.

        Args:
            num_samples (int): The number of samples to display.
        """
        
    def imshow(self, label: int, index: int):
        """
        Display an image from tha dataset with the corresponding label and index.
        
        Args:
            label (int): The label of the image.
            index (int): The index of the image.
        """


        