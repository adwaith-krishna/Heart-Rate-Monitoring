import numpy as np

def split_into_samples(images, labels, sample_size=(128, 128)):
    """Split large image and label arrays into smaller samples."""
    samples = []
    sample_labels = []

    img_height, img_width, _ = images.shape
    label_height, label_width = labels.shape
    sample_height, sample_width = sample_size

    for i in range(0, img_height, sample_height):
        for j in range(0, img_width, sample_width):
            if i + sample_height <= img_height and j + sample_width <= img_width:
                img_sample = images[i:i+sample_height, j:j+sample_width, :]
                label_sample = labels[i:i+sample_height, j:j+sample_width]

                samples.append(img_sample)
                sample_labels.append(label_sample)

    return np.array(samples), np.array(sample_labels)

def preprocess_and_save_samples(image_path, label_path, output_image_path, output_label_path):
    images = np.load(image_path)
    labels = np.load(label_path)

    # Normalize images to range [0, 1]
    images = images / 255.0

    # Ensure labels have a channel dimension
    if len(labels.shape) == 3:
        labels = np.expand_dims(labels, axis=-1)

    # Split into individual samples
    samples, sample_labels = split_into_samples(images, labels)

    print(f"Samples shape: {samples.shape}, Labels shape: {sample_labels.shape}")

    # Save the processed arrays
    np.save(output_image_path, samples)
    np.save(output_label_path, sample_labels)
    print(f"Processed samples saved to {output_image_path} and {output_label_path}")

if __name__ == "__main__":
    preprocess_and_save_samples(
        './datasets/skin/images.npy',
        './datasets/skin/labels.npy',
        './datasets/skin/samples_images.npy',
        './datasets/skin/samples_labels.npy'
    )
