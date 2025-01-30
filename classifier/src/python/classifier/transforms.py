"""
    transforms.py
    Author: Milan Marocchi
    
    Purpose: Create transforms for ml.
"""

from torchvision import transforms
import numpy as np

def get_pil_transform_numpy(size: int = 224) -> transforms.Compose:
    """
    Transform to get an image to show for xai but outputting a numpy image
    """

    transf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x.numpy()).astype(np.float64))  # Convert tensor to np array
    ])

    return transf


def get_pil_transform(size: int = 224) -> transforms.Compose:
    """
    Transform to get an image to show for xai
    """
    transf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    return transf


def get_normalise_transform(size: int = 224) -> transforms.Compose:
    """
    Applies the pre-processing transforms to the image as done for classification
    """
    transf = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transf


def get_preprocess_transform(size: int = 224) -> transforms.Compose:
    """
    Applies the pre-processing transforms to the image as done for classification
    """
    transf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transf


def create_data_transforms(is_inception: bool = False) -> dict[str, transforms.Compose]:
    """
    Creates data transforms for training and classifying
    """
    size = 299 if is_inception else 224

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    return data_transforms
