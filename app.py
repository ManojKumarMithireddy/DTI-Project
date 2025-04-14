from flask import Flask, render_template, jsonify, Response
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque
import time
import threading
import json
import os
from datetime import datetime
from openai import OpenAI

app = Flask(__name__)

# Initialize LM Studio client for mock_llm
client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
MODEL = "llama-3.2-1b-instruct"  # Adjust to your LM Studio model

# Load TFLite model and labels
interpreter = tf.lite.Interpreter(model_path='isl_model.tflite')  # Adjust path
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_scale, input_zero_point = input_details[0]['quantization']
output_scale, output_zero_point = output_details[0]['quantization']

labels = ['I', 'Eat', 'friend', 'go', 'Good', 'Happy', 'Hello', 'Love', 'what', 'n', 'Please', 'Thank you', 'want', 'who', 'Home', 'you']

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Global variables
sequence = deque(maxlen=30)
sentence = ""
history = []
last_pred_time = 0
pred_interval = 0.5
cap = None
running = False
lock = threading.Lock()
last_sentence_time = 0
pause_duration = 3
current_frame = None

# History file
HISTORY_FILE = 'sign_history.json'

def load_history():
    global history
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)

def save_history():
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

def normalize_landmarks(landmarks):
    wrist = landmarks[:3]
    wrist_tiled = np.tile(wrist, 21)
    normalized = landmarks - wrist_tiled
    scale = np.std(normalized) + 1e-6
    normalized /= scale
    return normalized

def mock_llm(sequence):
    """Generate a coherent sentence from a sequence of ISL signs using LM Studio."""
    if not sequence:
        return "No signs detected."
    
    # Prepare prompt for the language model
    signs_str = " ".join(sequence)
    prompt = (
        "Given a sequence of words, generate a grammatically correct and natural English sentence. "
        "The sequence may be incomplete or out of order, so infer the most likely meaning based on the words provided in their order. Try to make the sentence as coherent and meaningful as possible while maintaining the original words. "
        "Return ONLY the sentence itself, with no period, explanations, breakdowns, or additional text. "
        "Try to use all the words in the sequence in a meaningful way . "
        f"Sequence: {signs_str}"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant skilled in language interpretation."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.7
        )
        sentence = response.choices[0].message.content.strip()
        return signs_str+" : "+sentence if sentence else "Unable to interpret signs"
    except Exception as e:
        print(f"Error in mock_llm: {e}")
        # Fallback to simple joining if LM Studio fails
        return " ".join(sequence)

def video_processing():
    global sequence, sentence, history, last_pred_time, last_sentence_time, running, current_frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            frame = cv2.flip(frame, 1)
            with lock:
                current_frame = frame.copy()

            current_time = time.time()
            if current_time < last_sentence_time + pause_duration:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            landmarks = None
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                    landmarks = normalize_landmarks(landmarks)

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

                with lock:
                    if not sequence or pred_label != sequence[-1]:
                        sequence.append(pred_label)
                        if pred_label == 'n':
                            if sequence:
                                new_sentence = mock_llm(list(sequence)[:-1])
                                if new_sentence != "No signs detected.":
                                    sentence = new_sentence
                                    history.append({
                                        'sentence': sentence,
                                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    })
                                    save_history()
                                    last_sentence_time = current_time
                                sequence.clear()

            if sequence and len(sequence) == sequence.maxlen:
                with lock:
                    new_sentence = mock_llm(list(sequence))
                    if new_sentence != "No signs detected.":
                        sentence = new_sentence
                        history.append({
                            'sentence': sentence,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                        save_history()
                        last_sentence_time = current_time
                    sequence.clear()

    finally:
        cap.release()
        hands.close()

def generate_frames():
    while True:
        with lock:
            if current_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', current_frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    global running
    if not running:
        running = True
        threading.Thread(target=video_processing, daemon=True).start()
    return jsonify({'status': 'started'})

@app.route('/stop', methods=['POST'])
def stop():
    global running
    running = False
    return jsonify({'status': 'stopped'})

@app.route('/status')
def status():
    with lock:
        return jsonify({
            'sentence': sentence,
            'sequence': list(sequence),
            'history': history
        })

@app.route('/clear_history', methods=['POST'])
def clear_history():
    global history
    with lock:
        history = []
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
    return jsonify({'status': 'history cleared'})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    load_history()
    app.run(debug=True)