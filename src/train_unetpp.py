import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from data_loader import load_ubiris_dataset
from preprocess import augment_data
from utils import save_plots  # your plotting function

# ======================
# Configuration
# ======================
CONFIG = {
    'input_shape': (256, 256, 3),
    'batch_size': 8,
    'epochs': 50,
    'learning_rate': 1e-4,
    'validation_split': 0.1,
    'model_save_path': 'models/unetpp_iris.h5',
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5
}

# ======================
# U-Net++ Blocks
# ======================
def conv_block(x, filters):
    """Convolutional block with Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    return x

def build_unetpp(input_shape: tuple) -> tf.keras.Model:
    """Build U-Net++ model for iris segmentation"""
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder (downsampling path)
    x0_0 = conv_block(inputs, 64)
    x1_0 = conv_block(layers.MaxPool2D()(x0_0), 128)
    x2_0 = conv_block(layers.MaxPool2D()(x1_0), 256)
    x3_0 = conv_block(layers.MaxPool2D()(x2_0), 512)
    x4_0 = conv_block(layers.MaxPool2D()(x3_0), 1024)

    # Decoder (nested skip connections)
    x0_1 = conv_block(layers.Concatenate()([x0_0, layers.UpSampling2D()(x1_0)]), 64)
    x1_1 = conv_block(layers.Concatenate()([x1_0, layers.UpSampling2D()(x2_0)]), 128)
    x2_1 = conv_block(layers.Concatenate()([x2_0, layers.UpSampling2D()(x3_0)]), 256)
    x3_1 = conv_block(layers.Concatenate()([x3_0, layers.UpSampling2D()(x4_0)]), 512)

    x0_2 = conv_block(layers.Concatenate()([x0_0, x0_1, layers.UpSampling2D()(x1_1)]), 64)
    x1_2 = conv_block(layers.Concatenate()([x1_0, x1_1, layers.UpSampling2D()(x2_1)]), 128)
    x2_2 = conv_block(layers.Concatenate()([x2_0, x2_1, layers.UpSampling2D()(x3_1)]), 256)

    x0_3 = conv_block(layers.Concatenate()([x0_0, x0_1, x0_2, layers.UpSampling2D()(x1_2)]), 64)
    x1_3 = conv_block(layers.Concatenate()([x1_0, x1_1, x1_2, layers.UpSampling2D()(x2_2)]), 128)

    x0_4 = conv_block(layers.Concatenate()([x0_0, x0_1, x0_2, x0_3, layers.UpSampling2D()(x1_3)]), 64)

    # Output layer (binary segmentation)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x0_4)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

# ======================
# Loss and Metrics
# ======================
def dice_coef(y_true, y_pred, smooth=1e-6):
    """Dice coefficient metric"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )

def dice_loss(y_true, y_pred):
    """Dice loss function"""
    return 1 - dice_coef(y_true, y_pred)

# ======================
# Training Loop
# ======================
def train():
    # Load dataset
    print("Loading dataset...")
    X_train, X_test, y_train, y_test = load_ubiris_dataset(base_path='../dataset/ubiris')

    # Data augmentation
    print("Augmenting data...")
    X_train_aug, y_train_aug = augment_data(X_train, y_train)

    # Build model
    print("Building U-Net++ model...")
    model = build_unetpp(CONFIG['input_shape'])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss=dice_loss,
        metrics=[
            dice_coef,
            'accuracy',
            tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1], name='iou'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    # Callbacks
    callbacks_list = [
        callbacks.ModelCheckpoint(
            CONFIG['model_save_path'],
            monitor='val_dice_coef',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_dice_coef',
            patience=CONFIG['early_stopping_patience'],
            mode='max',
            verbose=1,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_dice_coef',
            factor=0.1,
            patience=CONFIG['reduce_lr_patience'],
            verbose=1,
            mode='max'
        ),
        callbacks.TensorBoard(log_dir='./logs')
    ]

    # Train model
    print("Training model...")
    history = model.fit(
        X_train_aug,
        y_train_aug,
        batch_size=CONFIG['batch_size'],
        epochs=CONFIG['epochs'],
        validation_data=(X_test, y_test),
        callbacks=callbacks_list,
        shuffle=True
    )

    # Save training curves
    save_plots(history)

    # Evaluate model
    print("Evaluating model...")
    results = model.evaluate(X_test, y_test, batch_size=CONFIG['batch_size'])
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Dice Coefficient: {results[1]:.4f}")
    print(f"Test Accuracy: {results[2]:.4f}")
    print(f"Test IoU: {results[3]:.4f}")
    print(f"Test Precision: {results[4]:.4f}")
    print(f"Test Recall: {results[5]:.4f}")

# ======================
# Main
# ======================
if __name__ == "__main__":
    # GPU memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Create folders
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    # Train
    train()
