import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import jaccard_score

# Function to load data
def load_test_data(image_path, label_path):
    images = np.load(image_path)
    labels = np.load(label_path)

    # Ensure labels have a channel dimension
    if len(labels.shape) == 3:
        labels = np.expand_dims(labels, axis=-1)

    return images, labels

# Function to evaluate the model
def evaluate_model_fixed(model_path, test_images, test_labels):
    # Load the trained model
    model = load_model(model_path, compile=False)  # Disable compilation initially

    # Compile the model with appropriate loss and metrics
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',  # Use binary_crossentropy for segmentation
                  metrics=['accuracy'])        # Include accuracy as a metric

    # Predict on test data
    print("Generating predictions on test data...")
    y_pred = model.predict(test_images)
    y_pred_binary = (y_pred > 0.5).astype(np.int32)  # Binarize predictions (threshold at 0.5)

    # Flatten predictions and labels for IoU calculation
    y_pred_flat = y_pred_binary.flatten()
    y_true_flat = test_labels.flatten()

    # Compute Jaccard (IoU) score
    print("Calculating IoU (Jaccard Score)...")
    iou_score = jaccard_score(y_true_flat, y_pred_flat, average='binary')
    print(f"Overall IoU (Jaccard Score): {iou_score:.4f}")

    # Evaluate using model metrics (loss and accuracy)
    print("Evaluating model metrics...")
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=1)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    return y_pred_binary

if __name__ == "__main__":
    # Paths to test data
    test_images_path = './datasets/skin/test_images.npy'
    test_labels_path = './datasets/skin/test_labels.npy'
    model_path = './models/skin_segmentation_model.h5'

    # Load test data
    test_images, test_labels = load_test_data(test_images_path, test_labels_path)
    print(f"Test images shape: {test_images.shape}")
    print(f"Test labels shape: {test_labels.shape}")

    # Evaluate the model
    predictions = evaluate_model_fixed(model_path, test_images, test_labels)

    # Save predictions as numpy array (optional)
    np.save('./predictions/skin_predictions.npy', predictions)
    print("Predictions saved successfully.")
