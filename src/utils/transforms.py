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

def inference_transform(image, device):
    """
    Transform for inference
    """
    return transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])(image).unsqueeze(0).to(device)

def optimized_transform(is_training=True):
    """
    Optimized transforms with better augmentation strategy
    """
    base_transforms = [
        transforms.Resize((256, 256)),  # Larger initial size for better cropping
        transforms.CenterCrop(224),     # Consistent center crop
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    
    if is_training:
        training_transforms = [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        ]
        return transforms.Compose(training_transforms + base_transforms)
    
    return transforms.Compose(base_transforms)