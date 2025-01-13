import tensorflow as tf
import numpy as np

def predict_heart_rate_lstm(signals, model_path="models/lstm_heart_rate_model.h5"):
    model = tf.keras.models.load_model(model_path)
    signals = np.expand_dims(signals, axis=0)  # Add batch dimension
    predicted_heart_rate = model.predict(signals)
    return predicted_heart_rate[0, 0]  # Extract the prediction
