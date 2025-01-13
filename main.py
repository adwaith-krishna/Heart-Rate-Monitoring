import cv2
from cnn_skin_detection import load_cnn_model, detect_skin_cnn
from lstm_heart_rate import predict_heart_rate_lstm
from video_capture import capture_webcam_frames
from signal_processing import extract_rgb_signals

def main():
    cnn_model = load_cnn_model()
    lstm_model_path = "models/lstm_heart_rate_model.h5"

    print("Press 'q' to quit.")
    cap = cv2.VideoCapture(0)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        skin = detect_skin_cnn(frame, cnn_model)
        cv2.imshow("Skin Segmentation", skin)
        frames.append(skin)

        if len(frames) == 30:  # Process every 30 frames
            signals = extract_rgb_signals(frames)
            heart_rate = predict_heart_rate_lstm(signals, lstm_model_path)
            print(f"Estimated Heart Rate: {heart_rate:.2f} BPM")
            frames = []

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
