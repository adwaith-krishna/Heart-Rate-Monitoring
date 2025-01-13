import tensorflow as tf
import cv2
import numpy as np

def load_cnn_model(model_path="models/skin_segmentation_model.h5"):
    return tf.keras.models.load_model(model_path)

def detect_skin_cnn(frame, model):
    resized_frame = cv2.resize(frame, (128, 128)) / 255.0
    mask = model.predict(np.expand_dims(resized_frame, axis=0))[0]
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    skin = cv2.bitwise_and(frame, frame, mask=(mask > 0.5).astype(np.uint8) * 255)
    return skin
