import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from data_loader import load_ubiris_dataset
from preprocess import augment_data
from utils import save_plots  # Using your existing function instead of plot_training_curve

# Configuration
CONFIG = {
    'input_shape': (256, 256, 3),
    'batch_size': 8,
    'epochs': 50,
    'learning_rate': 1e-4,
    'validation_split': 0.1,
    'model_save_path': 'models/unet_iris.h5',
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5
}

def build_unet(input_shape: tuple) -> tf.keras.Model:
    """Build U-Net model for iris segmentation"""
    inputs = tf.keras.Input(shape=input_shape)
    
    # Encoder
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    res1 = x
    x = layers.MaxPool2D()(x)
    
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    res2 = x
    x = layers.MaxPool2D()(x)
    
    # Bottleneck
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    
    # Decoder
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(128, 2, padding='same', activation='relu')(x)
    x = layers.Concatenate()([x, res2])
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, 2, padding='same', activation='relu')(x)
    x = layers.Concatenate()([x, res1])
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    
    # Output
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

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
    X_train, X_test, y_train, y_test = load_ubiris_dataset(base_path='../dataset/ubiris')
    
    # Data augmentation
    print("Augmenting data...")
    X_train_aug, y_train_aug = augment_data(X_train, y_train)
    
    # Build model
    print("Building model...")
    model = build_unet(CONFIG['input_shape'])
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
    
    # Save training curves using your existing function
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
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    train()