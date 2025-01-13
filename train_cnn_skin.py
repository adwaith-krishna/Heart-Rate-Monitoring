import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, UpSampling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU

# Load the dataset
def load_data(image_path, label_path):
    images = np.load(image_path)
    labels = np.load(label_path)

    # Ensure labels are binary and have a channel dimension
    labels = (labels > 0.5).astype(np.float32)  # Ensure binary labels
    if len(labels.shape) == 3:
        labels = np.expand_dims(labels, axis=-1)

    return images, labels

# Define the model architecture
def build_model():
    model = Sequential([
        Input(shape=(128, 128, 3)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(1, (1, 1), activation='sigmoid', padding='same')  # Output layer
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', MeanIoU(num_classes=2)])  # Ensure correct num_classes

    return model

# Train the model
def train_model(model, train_images, train_labels, val_images, val_labels, batch_size=16, epochs=20):
    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        batch_size=batch_size,
        epochs=epochs
    )
    return history

if __name__ == "__main__":
    # Paths to data
    train_images_path = './datasets/skin/train_images.npy'
    train_labels_path = './datasets/skin/train_labels.npy'
    val_images_path = './datasets/skin/test_images.npy'
    val_labels_path = './datasets/skin/test_labels.npy'

    # Load training and validation data
    train_images, train_labels = load_data(train_images_path, train_labels_path)
    val_images, val_labels = load_data(val_images_path, val_labels_path)

    # Debug: Check data shape and unique values
    print(f"Training images shape: {train_images.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Validation images shape: {val_images.shape}")
    print(f"Validation labels shape: {val_labels.shape}")
    print(f"Train labels unique values: {np.unique(train_labels)}")
    print(f"Validation labels unique values: {np.unique(val_labels)}")

    # Build and train the model
    model = build_model()
    history = train_model(model, train_images, train_labels, val_images, val_labels, batch_size=16, epochs=20)

    # Save the model
    model.save('./models/skin_segmentation_model.h5')
    print("Model saved successfully.")
