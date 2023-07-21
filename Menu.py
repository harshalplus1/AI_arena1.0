#!/usr/bin/env python
# coding: utf-8


import numpy as np
import cv2
import mediapipe as mp
import time
import subprocess
import sys

def choice(frame):
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    w, h = 640, 480
    cv2.rectangle(frame, (10, 10), (145, 55), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 90), (145, 135), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 170), (145, 215), (0, 0, 0), -1)
    cv2.rectangle(frame, (505, 10), (640, 55), (0, 0, 0), -1)
    cv2.rectangle(frame, (505, 90), (640, 135), (0, 0, 0), -1)
    cv2.rectangle(frame, (505, 170), (640, 215), (0, 0, 0), -1)

    cv2.putText(
        frame,
        "1.Biceps",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "2.BarbellRow",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "3.JumpingJacks",
        (20, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "4.FrontRaise",
        (515, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "5.ShoulderPress",
        (515, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "6.Squats",
        (515, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(frame, (315, 0), (360, 20), (0, 0, 0), -1)
    cv2.putText(
        frame,
        "Exit",
        (320, 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    results = mp_hands.process(frame_rgb)
    x_px, y_px = -1, -1
    if results.multi_hand_landmarks:
        # Get the first (right) hand's landmarks
        hand_landmarks = results.multi_hand_landmarks[0]

        # Get the right index finger coordinates (landmark 8)
        index_finger_landmark = hand_landmarks.landmark[
            mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP
        ]
        x_px, y_px = int(index_finger_landmark.x * w), int(index_finger_landmark.y * h)

    # Draw a circle at the right index finger's location
    cv2.circle(frame, (x_px, y_px), 8, (0, 255, 0), -1)
    return frame, x_px, y_px
        
    

def main():
    python_interpreter = f"{sys.executable}" 
    call_script = None
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        nframe, x, y = choice(frame)
        cv2.circle(nframe, (x, y), 8, (0, 255, 0), 2)
        if x > 10 and x < 145 and y > 10 and y < 55:
            call_script = "Biceps.py"
        elif x > 10 and x < 145 and y > 90 and y < 135:
            call_script = "BarbellRow.py"
        elif x > 10 and x < 145 and y > 170 and y < 215:
            call_script = "JumpingJacks.py"
        elif x > 505 and x < 640 and y > 10 and y < 55:
            call_script = "FrontRaise.py"
        elif x > 505 and x < 640 and y > 90 and y < 135:
            call_script = "ShoulderPress.py"
        elif x > 505 and x < 640 and y > 170 and y < 215:
            call_script = "Squats.py"
        elif x > 315 and x < 360 and y > 0 and y < 20:
            break
        command = [python_interpreter, call_script]
        if call_script is not None:
            try:
                subprocess.Popen(command)
                break
            except subprocess.CalledProcessError as e:
                print(f"Error executing the script: {e}")
            except FileNotFoundError:
                print("Enter the correct PATH.")
        cv2.imshow("Menu", nframe)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

