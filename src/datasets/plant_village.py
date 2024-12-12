"""
Plant Village Dataset Wrapper Class
"""

import os
from torch.utils.data import DataLoader

from torch.utils.data import Dataset, random_split
from torchvision import transforms, datasets

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

    def _default_transforms(self):
        """
        Default data augmentation and preprocessing transforms
        """
        return transforms.Compose([
            # convert to tensor and normalize using ImageNet stats
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            # resize images to consistent size
            transforms.Resize((224, 224)),

            # random augmentations for training
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
        ])
    
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

    # Create DataLoader for both
    train_loader = DataLoader(train, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val, batch_size=32, shuffle=False, num_workers=4)

    # Loop through the DataLoader
    for images, labels in train_loader:
        print(images.size(), labels)
