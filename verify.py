import numpy as np

images = np.load('./datasets/skin/samples_images.npy')
labels = np.load('./datasets/skin/samples_labels.npy')

print("Processed Images shape:", images.shape)
print("Processed Labels shape:", labels.shape)
