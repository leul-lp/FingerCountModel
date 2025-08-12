import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Setup webcam
cap = cv2.VideoCapture(0)

# Define the CSV file path
csv_file_path = 'hand_gesture_data_2.csv'

# Check if the CSV file already exists to avoid rewriting headers
file_exists = os.path.exists(csv_file_path)

# Open the CSV file in append mode
csv_file = open(csv_file_path, 'a', newline='')
csv_writer = csv.writer(csv_file)

# Write headers only if the file is new
if not file_exists:
    headers = [f'landmark_{i}_{coord}' for i in range(21) for coord in ['x', 'y', 'z']]
    headers.append('label')
    csv_writer.writerow(headers)
    print(f"CSV file '{csv_file_path}' created with headers.")
else:
    print(f"Appending to existing CSV file '{csv_file_path}'.")

# Main loop to capture and process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a more intuitive view
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    feedback_text = ""
    landmark_data = []

    # Check if a hand was detected
    if results.multi_hand_landmarks:
        # We only expect one hand due to max_num_hands=1
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Draw the landmarks on the frame
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extract the coordinates
        for landmark in hand_landmarks.landmark:
            landmark_data.extend([landmark.x, landmark.y, landmark.z])

    # Check for keyboard input to save data
    key = cv2.waitKey(1) & 0xFF
    
    # Check if a hand was detected before saving
    if landmark_data and key in [ord(str(i)) for i in range(6)]:
        # Convert key to a string for the label
        label = str(key - ord('0'))
        
        # Append the label and write the row
        landmark_data.append(label)
        csv_writer.writerow(landmark_data)
        
        # Set feedback text
        feedback_text = f"Gesture '{label}' saved!"
        print(f"Gesture '{label}' saved!")
    
    # Display feedback text on the screen
    cv2.putText(frame, feedback_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the resulting frame
    cv2.imshow('Hand Gesture Data Collector', frame)
    
    # Exit on 'q' press
    if key == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
csv_file.close()