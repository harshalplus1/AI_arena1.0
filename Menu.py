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
    cv2.rectangle(frame, (420, 1), (635, 45), (0, 0, 0), -1)
    cv2.rectangle(frame, (420, 60), (635, 105), (0, 0, 0), -1)
    cv2.rectangle(frame, (420, 115), (635, 165), (0, 0, 0), -1)
    cv2.rectangle(frame, (420, 175), (635, 225), (0, 0, 0), -1)
    cv2.rectangle(frame, (420, 235), (635, 285), (0, 0, 0), -1)
    cv2.rectangle(frame, (420, 295), (635, 345), (0, 0, 0), -1)
    cv2.rectangle(frame, (420, 355), (635, 405), (0, 0, 0), -1)
    cv2.putText(
        frame,
        "1.Biceps",
        (425, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "2.Shoulders",
        (425, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "3.BarbellRow",
        (425, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "4.JumpingJacks",
        (425, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "5.FrontRaise",
        (425, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "6.ShoulderPress",
        (425, 320),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "7.Squats",
        (425, 380),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
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
    python_interpreter = f"{sys.executable}"  # Change this if your Python executable has a different name or path
    call_script = None
    print(python_interpreter)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        nframe, x, y = choice(frame)
        option = None
        project_launched = None
        cv2.circle(nframe, (x, y), 8, (0, 255, 0), 2)
        if x > 410 and x < 440 and y > 5 and y < 35:
            cv2.putText(
                frame,
                "1.Biceps",
                (425, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            call_script = "Biceps.py"
        elif x > 410 and x < 440 and y > 60 and y < 90:
            cv2.putText(
                frame,
                "2.FrontRaise",
                (425, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            call_script = "FrontRaise.py"
        elif x > 410 and x < 440 and y > 115 and y < 145:
            cv2.putText(
                frame,
                "3.BarbellRow.py",
                (425, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            call_script = "BarbellRow.py"
        elif x > 410 and x < 440 and y > 170 and y < 210:
            cv2.putText(
                frame,
                "4.JumpingJacks.py",
                (425, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            call_script = "JumpingJacks.py"
        elif x > 410 and x < 440 and y > 230 and y < 270:
            cv2.putText(
                frame,
                "5.FrontRaise.py",
                (425, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            call_script = "FrontRaise.py"
        elif x > 410 and x < 440 and y > 290 and y < 330:
            cv2.putText(
                frame,
                "6.ShoulderPress.py",
                (425, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            call_script = "ShoulderPress.py"
        elif x > 410 and x < 440 and y > 350 and y < 390:
            cv2.putText(
                frame,
                "7.Squats.py",
                (425, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            call_script = "Squats.py"
        command = [python_interpreter, call_script]
        if call_script is not None:
            try:
                subprocess.Popen(command)
                break
            except subprocess.CalledProcessError as e:
                print(f"Error executing the script: {e}")
            except FileNotFoundError:
                print("Enter the correct PATH.")
        cv2.imshow("Output", nframe)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

