import mediapipe as mp
import cv2 as cv
import time as t
import sys 

import torch
import torch.nn.functional as F


import joblib
from data.model.sign_model_arch import SignModel


capture = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    # Default Params
    max_num_hands = 1
)

mpDraw = mp.solutions.drawing_utils
cTime = 0
pTime = 0

scaler = joblib.load('./data/model/scaler.pkl')

new_model = SignModel()
new_model.load_state_dict(torch.load('./data/model/SignModel.pth'))


def predict(landmarks):


    # Get the landmarks
    landmarks_list = []

    if landmarks.landmark:
        for lm in landmarks.landmark:
            landmarks_list.append(lm.x)
            landmarks_list.append(lm.y)
            landmarks_list.append(lm.z)

    landmarks_tensor = torch.tensor(landmarks_list, dtype=torch.float32).reshape(1, -1)

    # Process the landmarks
    try:
        scaled_landmark = scaler.transform(landmarks_tensor.numpy())
    except Exception as e:
        print(f"Error during scaling: {e}")
        return None
    
    tensor_data = torch.from_numpy(scaled_landmark).type(torch.float32)

    # make a prediction
    with torch.inference_mode():
        output = new_model(tensor_data)
        probs = F.softmax(output)

        predicted = torch.argmax(probs)

        return predicted

while True:
    _, frame = capture.read()

    frame = cv.flip(frame, 1)
    rgbImage = cv.cvtColor(frame, cv.COLOR_BGR2RGB)



    results = hands.process(rgbImage)
    
    multi_hand_landmarks, multi_handedness = results.multi_hand_landmarks, results.multi_handedness

    if multi_hand_landmarks and multi_handedness:
        for hand_landmark, multi_handedness in zip(multi_hand_landmarks, multi_handedness):
            mpDraw.draw_landmarks(
                    frame, 
                    hand_landmark, 
                    mpHands.HAND_CONNECTIONS, 
                    landmark_drawing_spec=mpDraw.DrawingSpec(color=(0, 0, 0), thickness=3, circle_radius=3),
                    connection_drawing_spec = mpDraw.DrawingSpec(color=(255, 255, 255), thickness=5)
                )
            
            # Draw the box around the hands
            h, w, c = frame.shape
            x_min, y_min = h, w
            x_max, y_max = 0, 0

            padding = 0
            

            for lm in hand_landmark.landmark:
                x, y = int(lm.x * w), int(lm.y * h)

                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y


            prediction = predict(hand_landmark)


            # Drawing board (min coor) (max coor) (color) (thickness)
            cv.rectangle(frame, (x_min-padding, y_min-padding), (x_max+padding, y_max+padding), (100, 0, 255), 5)
            cv.putText(frame, f"Prediction {prediction}", (10, 100), cv.FONT_HERSHEY_SIMPLEX, 1,  (255, 255, 255), 2, cv.LINE_AA)

            hand_type = multi_handedness.classification[0].label
            cv.putText(frame, f"{hand_type} {prediction}", (x_min - padding, y_max + padding + 30), cv.FONT_HERSHEY_SIMPLEX, 1,  (255, 255, 255), 2, cv.LINE_AA)
    
    cTime = t.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(frame, f"FPS {str(int(fps))}", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1,  (0, 255, 0), 2, cv.LINE_AA)
    cv.imshow("Webcam preview", frame) 
    if cv.waitKey(1) & 0xFF == ord('q'):
        sys.exit(0)
