import numpy as np
from verify-cnn-model import verify_model  # Assuming verify_model is defined

# Load the sample data
sample_images = np.load('./datasets/skin/samples_images.npy')
sample_labels = np.load('./datasets/skin/samples_labels.npy')

# Path to the trained model
model_path = './models/skin_segmentation_model.h5'

# Verify the model
verify_model(model_path, sample_images, sample_labels)
