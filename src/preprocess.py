import cv2
import numpy as np
from typing import Optional, Tuple  # ✅ This now actually imports Tuple
import albumentations as A


def augment_data(images: np.ndarray, masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply data augmentation to images and masks using albumentations"""
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    ])
    
    augmented_images = []
    augmented_masks = []
    
    for img, mask in zip(images, masks):
        transformed = transform(image=img, mask=mask)
        augmented_images.append(transformed['image'])
        augmented_masks.append(transformed['mask'])
    
    return np.array(augmented_images), np.array(augmented_masks)

class IrisPreprocessor:
    def __init__(self, clahe_clip: float = 2.0, grid_size: Tuple[int, int] = (8, 8)):
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=grid_size
        )

    def remove_reflections(self, image: np.ndarray) -> np.ndarray:
        """Remove specular reflections from iris image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Full preprocessing pipeline"""
        inpainted = self.remove_reflections(image)
        enhanced = self.clahe.apply(inpainted)
        return enhanced.astype(np.float32) / 255.0

    @staticmethod
    def load_mask(path: str) -> Optional[np.ndarray]:
        """Load and validate mask"""
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return None if mask is None else (mask > 0).astype(np.uint8)
