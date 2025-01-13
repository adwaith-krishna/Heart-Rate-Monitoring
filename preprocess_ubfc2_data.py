import os
import cv2
import numpy as np

def preprocess_ubfc2_data(dataset_dir, output_dir, max_frames=1550):
    os.makedirs(output_dir, exist_ok=True)

    signals = []
    labels = []

    for subject in sorted(os.listdir(dataset_dir)):
        subject_dir = os.path.join(dataset_dir, subject)
        if not os.path.isdir(subject_dir):
            continue

        video_file = os.path.join(subject_dir, "vid.avi")
        label_file = os.path.join(subject_dir, "ground_truth.txt")

        if not os.path.exists(video_file) or not os.path.exists(label_file):
            print(f"Skipping {subject}: Missing video or label file")
            continue

        # Extract video frames
        cap = cv2.VideoCapture(video_file)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        subject_signals = []
        for _ in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, (64, 64))
            subject_signals.append(resized_frame)
        cap.release()

        # Pad or truncate frames
        if len(subject_signals) > max_frames:
            subject_signals = subject_signals[:max_frames]
        elif len(subject_signals) < max_frames:
            padding = [np.zeros((64, 64)) for _ in range(max_frames - len(subject_signals))]
            subject_signals.extend(padding)

        # Normalize the video frames
        subject_signals = np.array(subject_signals) / 255.0
        signals.append(subject_signals)

        # Process labels
        label_data = np.loadtxt(label_file)
        if label_data.ndim == 2:  # Flatten 2D labels into 1D if necessary
            label_data = label_data.flatten()
        if len(label_data) > max_frames:
            label_data = label_data[:max_frames]
        elif len(label_data) < max_frames:
            padding = np.zeros(max_frames - len(label_data))
            label_data = np.concatenate((label_data, padding))
        labels.append(label_data)

    # Convert to numpy arrays
    signals = np.array(signals)
    labels = np.array(labels, dtype=object)  # Use object dtype for variable-length data

    # Split into training and testing
    split_idx = int(0.8 * len(signals))
    train_signals, test_signals = signals[:split_idx], signals[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]

    # Save preprocessed data
    np.save(os.path.join(output_dir, "train_signals.npy"), train_signals)
    np.save(os.path.join(output_dir, "train_labels.npy"), train_labels, allow_pickle=True)
    np.save(os.path.join(output_dir, "test_signals.npy"), test_signals)
    np.save(os.path.join(output_dir, "test_labels.npy"), test_labels, allow_pickle=True)
    print("Preprocessing complete. Data saved to:", output_dir)

if __name__ == "__main__":
    preprocess_ubfc2_data("./datasets/UBFC2", "./datasets/UBFC2/preprocessed/")
