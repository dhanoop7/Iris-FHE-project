import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB0
from data_loader import load_ubiris_dataset
from preprocess import augment_data
from utils import save_plots

# Configuration
CONFIG = {
    'input_shape': (256, 256, 3),
    'batch_size': 8,
    'epochs': 50,
    'learning_rate': 1e-4,
    'validation_split': 0.1,
    'model_save_path': 'models/efficientunet_iris.h5',
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5
}

def build_efficientunet(input_shape: tuple) -> tf.keras.Model:
    """Build EfficientNet-based U-Net (EfficientUNet)"""
    # Encoder (EfficientNet)
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=tf.keras.Input(shape=input_shape)
    )

    # Choose layers for skip connections
    skips = [
        base_model.get_layer("block2a_expand_activation").output,   # 64x64
        base_model.get_layer("block3a_expand_activation").output,   # 32x32
        base_model.get_layer("block4a_expand_activation").output,   # 16x16
        base_model.get_layer("block6a_expand_activation").output    # 8x8
    ]
    encoder_output = base_model.get_layer("top_activation").output  # 8x8 bottleneck

    # Decoder
    x = encoder_output
    for skip in reversed(skips):
        x = layers.Conv2DTranspose(256, (3,3), strides=(2,2), padding="same")(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)

    # Final upsampling to match input size
    x = layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding="same")(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)

    # Output mask
    outputs = layers.Conv2D(1, (1,1), activation="sigmoid")(x)

    model = models.Model(inputs=base_model.input, outputs=outputs, name="EfficientUNet")
    return model

def dice_coef(y_true, y_pred, smooth=1e-6):
    """Dice coefficient metric"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice loss function"""
    return 1 - dice_coef(y_true, y_pred)

def train():
    # Load dataset
    print("Loading dataset...")
    X_train, X_test, y_train, y_test = load_ubiris_dataset(base_path="../dataset/ubiris")

    # Data augmentation
    print("Augmenting data...")
    X_train_aug, y_train_aug = augment_data(X_train, y_train)

    # Build model
    print("Building EfficientUNet model...")
    model = build_efficientunet(CONFIG['input_shape'])
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

if __name__ == "__main__":
    # Set memory growth to avoid GPU memory issues
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Create required directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    train()
