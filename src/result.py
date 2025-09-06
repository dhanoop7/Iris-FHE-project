import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.backend as K
from data_loader import load_ubiris_dataset  # Your existing data loader

# Define metrics and loss again
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# Load dataset
_, X_test, _, y_test = load_ubiris_dataset(base_path='../dataset/ubiris')

# Load model
model = tf.keras.models.load_model(
    'models/unetpp_iris.h5',
    custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef}
)

# Pick a sample
idx = 0
image = X_test[idx]
true_mask = y_test[idx]

# Predict
pred_mask = model.predict(np.expand_dims(image, axis=0))[0]
pred_mask_binary = (pred_mask.squeeze() > 0.5).astype(np.uint8)

# Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Input Image")

plt.subplot(1, 3, 2)
plt.imshow(true_mask, cmap='gray')
plt.title("Ground Truth")

plt.subplot(1, 3, 3)
plt.imshow(pred_mask_binary, cmap='gray')
plt.title("Predicted Mask")
plt.tight_layout()
plt.show()
