import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset(sample_images_path, sample_labels_path, test_size=0.1, random_state=42):
    # Load the sample data
    images = np.load('./datasets/skin/samples_images.npy')
    labels = np.load('./datasets/skin/samples_labels.npy')

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=random_state
    )

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")

    # Save the split datasets
    np.save('./datasets/skin/train_images.npy', X_train)
    np.save('./datasets/skin/train_labels.npy', y_train)
    np.save('./datasets/skin/test_images.npy', X_test)
    np.save('./datasets/skin/test_labels.npy', y_test)
    print("Datasets saved: train_images.npy, train_labels.npy, test_images.npy, test_labels.npy")

if __name__ == "__main__":
    split_dataset(
        './datasets/skin/samples_images.npy',
        './datasets/skin/samples_labels.npy',
        test_size=0.1  # Use 10% of the data for testing
    )
