import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque
import time

# Load TFLite model and labels
interpreter = tf.lite.Interpreter(model_path='isl_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_scale, input_zero_point = input_details[0]['quantization']
output_scale, output_zero_point = output_details[0]['quantization']

labels = ['I', 'Eat', 'friend', 'go', 'Good', 'Happy', 'Hello', 'Love', 'what', 'n', 'Please', 'Thank you', 'want', 'who','Home','you']  # 15 ISL signs
# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def normalize_landmarks(landmarks):
    """Normalize landmarks relative to wrist (landmark 0)"""
    wrist = landmarks[:3]  # x/y/z of wrist
    wrist_tiled = np.tile(wrist, 21)  # Repeat for 21 landmarks
    normalized = landmarks - wrist_tiled
    scale = np.std(normalized) + 1e-6
    normalized /= scale
    return normalized

def mock_llm(sequence):
    """Mock LLM: Join signs into a sentence"""
    if not sequence:
        return "No signs detected."
    return " ".join(sequence) + "."

def main():
    cap = cv2.VideoCapture(0)  # Webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    sequence = deque(maxlen=30)  # Buffer for signs
    sentence = ""
    last_pred_time = 0
    pred_interval = 0.5  # Predict every 0.5s to stabilize

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            landmarks = None
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                    landmarks = normalize_landmarks(landmarks)

            # Predict every 0.5s if landmarks are detected
            current_time = time.time()
            pred_label = None
            if landmarks is not None and (current_time - last_pred_time) >= pred_interval:
                input_data = landmarks.reshape(1, 63).astype(np.float32)
                input_data = (input_data / input_scale + input_zero_point).astype(np.int8)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
                pred_idx = np.argmax(output_data)
                pred_label = labels[pred_idx]
                last_pred_time = current_time

                # Buffer predictions
                if pred_label != sequence[-1] if sequence else True:  # Avoid duplicates
                    sequence.append(pred_label)
                    if pred_label == 'n':
                        sentence = mock_llm(list(sequence)[:-1])  # Exclude 'n'
                        sequence.clear()

            # Display info
            cv2.putText(frame, f"Sign: {pred_label or 'None'}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Sequence: {list(sequence)}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Sentence: {sentence}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow('ISL Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

if __name__ == "__main__":
    main()