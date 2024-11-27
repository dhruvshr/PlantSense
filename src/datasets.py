"""
File: datasets.py

Description: This module contains an abstract base class for datasets used in this project.
"""

import os
import pandas as pd
import cv2
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod

class Dataset(ABC):
    """
    An abstract base class for datasets used in this project.

    Args:
        - name: dataset name
        - path: path to the dataset
        - classes: list of classes
        - features: list of features
        - size: dataset size
    """
    def __init__(
            self,
            name: str,
            path: str,
            classes: List[str],
            features: List[str],
            size: int,
            general: bool) -> None:
        """
        Initializes the dataset object.
        """
        self.name = name
        self.path = path
        self.classes = classes
        self.features = features
        self.size = size
        self.general = general

    def load_data() -> None:
        """
        Loads the dataset.
        """
        pass

    def get_df() -> pd.DataFrame:
        """
        Returns the dataset in a pandas dataframe.
        """
        pass
    
    def get_feature_names() -> List[str]:
        """
        Returns the feature names.
        """
        pass

    def get_class_names() -> List[str]:
        """
        Returns the class names.
        """
        pass

    # def num_instances() -> int:
    #     """
    #     Returns the number of instances in the dataset.
    #     """
    #     pass

    # def num_classes() -> int:
    #     """
    #     Returns the number of classes in the dataset.
    #     """
    #     pass

    # def num_features() -> int:
    #     """
    #     Returns the number of features in the dataset.
    #     """
    #     pass


