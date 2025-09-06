import cv2
import numpy as np
from typing import Optional, Union
import tensorflow as tf
from pathlib import Path
from preprocess import IrisPreprocessor

class IrisSegmenter:
    def __init__(self, model_path: str = "models/best.h5"):
        self.model = self._load_model(model_path)
        self.preprocessor = IrisPreprocessor()

    def _load_model(self, path: str):
        try:
            return tf.keras.models.load_model(path)
        except Exception as e:
            raise ValueError(f"Model loading failed: {str(e)}")

    def _prepare_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """Handle both file paths and numpy arrays"""
        if isinstance(image, str):
            if not Path(image).exists():
                raise FileNotFoundError(f"Image not found: {image}")
            image = cv2.imread(image)
        
        processed = self.preprocessor.preprocess(image)
        resized = cv2.resize(processed, (256, 256))
        return resized[np.newaxis, ..., np.newaxis]  # Add batch and channel dims

    def segment(self, image: Union[str, np.ndarray], 
               threshold: float = 0.5) -> Optional[np.ndarray]:
        """Run segmentation on input"""
        try:
            prepared = self._prepare_image(image)
            pred = self.model.predict(prepared)[0]
            return (pred > threshold).astype(np.uint8) * 255
        except Exception as e:
            print(f"Segmentation failed: {str(e)}")
            return None

    def save_result(self, image_path: str, output_path: str):
        """Complete pipeline with saving"""
        mask = self.segment(image_path)
        if mask is not None:
            cv2.imwrite(output_path, mask)
            return True
        return False

if __name__ == "__main__":
    segmenter = IrisSegmenter()
    result = segmenter.save_result(
        "test_image.jpg",
        "output/segmentation.png"
    )
    print("Success!" if result else "Failed")