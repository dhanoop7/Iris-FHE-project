import os
import cv2
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split

def load_ubiris_dataset(base_path: str = 'dataset/ubiris',
                        target_size: Tuple[int, int] = (256, 256),
                        test_size: float = 0.2,
                        random_state: int = 42):

    images = []
    masks = []

    for subject_id in os.listdir(base_path):
        subject_path = os.path.join(base_path, subject_id)

        if not os.path.isdir(subject_path):
            continue

        image_dir = os.path.join(subject_path, 'images')
        mask_dir = os.path.join(subject_path, 'mask')

        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            print(f"⚠️ Skipping {subject_id}: missing 'images' or 'mask' folder")
            continue

        for img_file in sorted(os.listdir(image_dir)):
            if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(image_dir, img_file)
                mask_filename = os.path.splitext(img_file)[0] + '.png'
                mask_path = os.path.join(mask_dir, mask_filename)

                if os.path.exists(mask_path):
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, target_size)

                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask, target_size)
                    mask = (mask > 0).astype(np.float32)

                    images.append(img)
                    masks.append(mask)
                else:
                    print(f"⚠️ Mask not found for {img_file} in {subject_id}")

    if len(images) == 0:
        raise RuntimeError("No image-mask pairs found!")

    images = np.array(images, dtype=np.float32) / 255.0
    masks = np.array(masks, dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        images, masks, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test
