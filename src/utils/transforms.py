"""
Transforms
"""

from torchvision import transforms, datasets

def default_transform():
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

# def inference_transform():
