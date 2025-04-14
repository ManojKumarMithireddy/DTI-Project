import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import time

# Configuration
DATA_PATH = 'hand_sign_data'
ACTIONS = ['hello', 'please', 'thank_you', 'friend', 'love', 'eat', 'go', 'want', 'happy', 'good', 'what', 'who', 'I', 'you', 'home', 'n']  # 15 ISL signs
NUM_SEQUENCES = 30  # Number of videos per action
SEQUENCE_LENGTH = 30  # Frames per video
COLLECT_DELAY = 3  # Seconds between sequences within a sign

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def setup_folders():
    """Create folders for each action"""
    for action in ACTIONS:
        for seq in range(NUM_SEQUENCES):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(seq)))
            except FileExistsError:
                pass

def get_landmarks(frame):
    """Extract hand landmarks using MediaPipe"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
            )
            
            # Get normalized landmark coordinates
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        
        return frame, np.array(landmarks)
    return frame, None

def collect_data():
    """Main data collection function"""
    setup_folders()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access camera")
        return

    try:
        for action_idx, action in enumerate(ACTIONS):
            print(f'\n>>> Collecting data for: {action.upper()} <<<')
            print(f'Prepare to perform the "{action}" sign (check https://indiansignlanguage.org)...')
            
            # For first sign, no 'n' press needed; for others, wait for 'n'
            if action_idx > 0:
                while True:
                    ret, frame = cap.read()
                    if not ret: continue
                    cv2.putText(frame, f'Press "n" to start collecting for "{action}"...', 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Hand Sign Collector', frame)
                    key = cv2.waitKey(10) & 0xFF
                    if key == ord('n'):
                        break
                    elif key == ord('q'):
                        raise KeyboardInterrupt
            
            # Countdown before starting
            for countdown in range(3, 0, -1):
                ret, frame = cap.read()
                if not ret: continue
                
                cv2.putText(frame, f'Starting "{action}" in {countdown}...', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Hand Sign Collector', frame)
                cv2.waitKey(1000)
            
            # Collect sequences
            for seq in range(NUM_SEQUENCES):
                print(f'\nSequence {seq+1}/{NUM_SEQUENCES} - Perform the "{action}" sign now!')
                sequence_data = []
                frame_num = 0
                
                while frame_num < SEQUENCE_LENGTH:
                    ret, frame = cap.read()
                    if not ret: continue
                    
                    # Get landmarks
                    frame, landmarks = get_landmarks(frame)
                    
                    if landmarks is not None:
                        sequence_data.append(landmarks)
                        frame_num += 1
                        
                        # Visual feedback for recording
                        cv2.putText(frame, f'Recording: {action}', (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f'Sequence: {seq+1}/{NUM_SEQUENCES}', (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f'Frame: {frame_num}/{SEQUENCE_LENGTH}', (10, 110),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        # Wait for hand detection
                        cv2.putText(frame, 'Waiting for hand detection...', 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    cv2.imshow('Hand Sign Collector', frame)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        raise KeyboardInterrupt
                
                # Save sequence
                if len(sequence_data) == SEQUENCE_LENGTH:
                    for frame_num, landmarks in enumerate(sequence_data):
                        np.save(
                            os.path.join(DATA_PATH, action, str(seq), f"{frame_num}"),
                            landmarks
                        )
                else:
                    print(f"Error: Collected {len(sequence_data)} frames for sequence {seq+1}, expected {SEQUENCE_LENGTH}")
                    break  # Stop if sequence is incomplete
                
                # Short pause between sequences
                if seq < NUM_SEQUENCES - 1:
                    print(f"Get ready for next sequence in {COLLECT_DELAY} seconds...")
                    time.sleep(COLLECT_DELAY)
    
    except KeyboardInterrupt:
        print("\nData collection stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nData collection complete! Saved to: {os.path.abspath(DATA_PATH)}")

if __name__ == "__main__":
    print("""
    ISL HAND SIGN DATA COLLECTOR
    ---------------------------
    Instructions:
    1. Use one hand only
    2. Hold each sign steady (static pose) for recording
    3. Keep hand centered and well-lit in camera view
    4. If no hand is detected, collection pauses until hand appears
    5. After each sign, press 'n' to start the next sign
    6. Press 'Q' to quit early
    7. Check sign videos at https://indiansignlanguage.org for accuracy
    """)
    
    collect_data()