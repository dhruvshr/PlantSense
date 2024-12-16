"""
Plant Village Dataset Wrapper Class
"""

import os
from torch.utils.data import DataLoader

from torch.utils.data import Dataset, random_split
from torchvision import transforms, datasets
from src.utils.transforms import default_transform

PLANTVILLAGE_PATH = "data/raw/Plant_leave_diseases_dataset_with_augmentation"

class PlantVillage(Dataset):
    def __init__(self, dataset=PLANTVILLAGE_PATH, transform=None):
        """
        Custom Dataset Wrapper for Plant Village Dataset

        Args:
        - dataset (str): plant_village dataset path
        - transform (torchvision.transforms): data transformation to apply
        """
        self.dataset = dataset
        self.transform = transform or self._default_transforms()

        self.plantvillage = datasets.ImageFolder(self.dataset, transform=self.transform)
        self.NUM_CLASSES = len(self.plantvillage.classes)

    def _default_transforms(self):
        """
        Default data augmentation and preprocessing transforms
        """
        return default_transform()
    
    def __len__(self):
        return len(self.plantvillage)
    
    def __getitem__(self, idx):
        image, label = self.plantvillage[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def split_dataset(self, train_ratio=0.7, val_ratio=0.15):
        """
        Split dataset into train, validation, and test sets
        
        Args:
            dataset: Original dataset
            train_ratio (float): Proportion of training data
            val_ratio (float): Proportion of validation data

        Returns:
            Tuple of train, validation, test datasets
        """
        dataset_size = len(self.plantvillage)
        train_size = int(dataset_size * train_ratio)
        val_size = int(dataset_size * val_ratio)
        test_size = dataset_size - train_size - val_size
        
        return random_split(self.plantvillage, [train_size, val_size, test_size])
    



# main test driver
if __name__ == "__main__":

    if not os.path.exists(PLANTVILLAGE_PATH):
        raise FileNotFoundError(f"The specified dataset path does not exist: {PLANTVILLAGE_PATH}")
    else:
        print(f"Dataset path found: {PLANTVILLAGE_PATH}")


    # Example usage
    plant_village = PlantVillage()

    # Get training and validation splits
    train, val, test = plant_village.split_dataset()

    classes = plant_village.plantvillage.classes

    # plantvillage classes
    for cls in classes:
        print(cls)

    # # Create DataLoader for both
    # train_loader = DataLoader(train, batch_size=32, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val, batch_size=32, shuffle=False, num_workers=4)

    # # Loop through the DataLoader
    # for images, labels in train_loader:
    #     print(images.size(), labels)
