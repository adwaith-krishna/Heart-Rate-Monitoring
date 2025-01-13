import numpy as np
import tensorflow as tf
from lstm_heart_rate import build_lstm_model

def load_ubfc_data(signals_path, labels_path):
    signals = np.load(signals_path)
    labels = np.load(labels_path)
    return signals, labels

def train_heart_rate_model(signals_path, labels_path, output_model_path):
    signals, labels = load_ubfc_data(signals_path, labels_path)
    model = build_lstm_model((signals.shape[1], signals.shape[2]))
    model.fit(signals, labels, epochs=20, batch_size=32, validation_split=0.2)
    model.save(output_model_path)
    print(f"LSTM heart rate model saved to {output_model_path}")

if __name__ == "__main__":
    train_heart_rate_model('./datasets/UBFC2/signals.npy', './datasets/UBFC2/labels.npy', './models/lstm_heart_rate_model.h5')
