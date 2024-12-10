"""
file: dataset_interface.py
"""

from abc import ABC, abstractmethod
import cv2
from matplotlib import pyplot as plt
import os

class DatasetInterface(ABC):
    """
    Abstract base class for datasets. This interface handles loading and processing
    of data for training and testing.

    Methods:
        load_data(): Abstract method to load the dataset (e.g., images, CSV).
        get_classes() -> list[str]: Abstract method to return class labels.
        classes_to_idx() -> dict[str, int]: Abstract method to map class labels to indices.
        transform(): Optional method for data transformations (resize, normalize).
        get_data(): Helper method to retrieve the data and labels.
    """

    def __init__(self, name: str, path: str, augmented: bool):
        """
        Initialize the dataset interface with the path to the dataset.

        Args:
            path (str): Path to the dataset (directory or file).
        """
        self.name = name
        self.path = path
        self.data = None    
        self.labels = None
        self.classes = None
        self.augmented = augmented
    
    @abstractmethod
    def load_data(self, batch_size: int = 32, shuffle: bool = True):
        """
        Load data from the dataset (e.g., read images from disk, load CSV).
        The data and labels will be stored in `self.data` and `self.labels`.
        """
        pass

    @abstractmethod
    def get_classes(self) -> list[str]:
        """
        Get the list of class labels in the dataset.
        """
        pass

    @abstractmethod
    def get_classes_idx(self) -> dict[str, int]:
        """
        Convert class labels to class indices.
        """
        pass

    def head(self, num_samples: int = 5):
        """
        Prints the first few data samples and labels.
        """
        if self.data is not None and self.labels is not None:
            for i in range(min(num_samples, len(self.data))):
                print(f"Sample {i}: {self.data[i]}, Label: {self.labels[i]}")
        else:
            print("Data or labels not loaded yet.")

    def transform(self):
        """
        Apply transformations to the data (e.g., resize, normalize).
        This can be overridden by subclasses if specific transformations are needed.
        """
        pass
    
    def get_data(self):
        """
        Return the loaded data and labels.
        Useful for interfacing with PyTorch's DataLoader.
        
        Returns:
            tuple: (data, labels)
        """
        return self.data, self.labels

    def imshow(self, label: int, index: int):
        """
        Display an image from the dataset at the specified label (integer) and index.

        Args:
            label (int): Class label (integer index of the class).
            index (int): Index of the image in the class folder.
        """
        if label < 0 or label >= len(self.classes):
            print(f"Invalid label: {label}. It should be between 0 and {len(self.classes)-1}.")
            return
        
        # Get the folder for the specified label (class)
        class_name = self.classes[label]
        class_folder = os.path.join(self.path, class_name)
        
        # Check if the class folder exists
        if not os.path.isdir(class_folder):
            print(f"Class folder '{class_folder}' does not exist.")
            return

        # List all images in the class folder
        image_files = os.listdir(class_folder)
        
        # Check if the index is valid
        if index < 0 or index >= len(image_files):
            print(f"Index {index} is out of bounds for class '{class_name}'.")
            return
        
        # Get the image path
        image_path = os.path.join(class_folder, image_files[index])
        
        # Read the image
        img = cv2.imread(image_path)

        if img is None:
            print(f"Failed to load image at {image_path}")
            return
        
        # Convert BGR (OpenCV default) to RGB for display (if needed)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display the image using Matplotlib
        plt.imshow(img_rgb)
        plt.title(f"Class: {class_name} (Label: {label}), Index: {index}")
        plt.axis('off')  # Hide axes
        plt.show()

        # Optionally, print the class label to the console
        print(f"Showing image from class: '{class_name}' with label: {label}")

    