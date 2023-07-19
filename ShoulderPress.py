#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cv2
import mediapipe as mp
from Angle import calculate_angle
import subprocess
import sys


# In[3]:


mp_drawing = mp.solutions.drawing_utils


# In[4]:


def shoulderpressutils(frame):
    # Convert frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Pose model
    results = pose.process(frame_rgb)

    # Check if any poses are detected
    if results.pose_landmarks:
        # Get landmark coordinates
        landmarks = results.pose_landmarks.landmark
        w, h = 640, 480
        # Specify LEFT shoulder, elbow, and wrist landmark indices
        l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        l_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        lsh = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
        ]
        lel = [
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
        ]
        lwr = [
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
        ]
        # Specify RIGHT shoulder, elbow, and wrist landmark indices
        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        r_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        rsh = [
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
        ]
        rel = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
        ]
        rwr = [
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
        ]
        # Get frame dimensions

        # Convert LEFT landmark positions to pixel coordinates
        lshoulder_x, lshoulder_y = int(l_shoulder.x * w), int(l_shoulder.y * h)
        lelbow_x, lelbow_y = int(l_elbow.x * w), int(l_elbow.y * h)
        lwrist_x, lwrist_y = int(l_wrist.x * w), int(l_wrist.y * h)
        langle = calculate_angle(lsh, lel, lwr)
        # Convert RIGHT landmark positions to pixel coordinates
        rshoulder_x, rshoulder_y = int(r_shoulder.x * w), int(r_shoulder.y * h)
        relbow_x, relbow_y = int(r_elbow.x * w), int(r_elbow.y * h)
        rwrist_x, rwrist_y = int(r_wrist.x * w), int(r_wrist.y * h)
        rangle = calculate_angle(rsh, rel, rwr)
        cv2.line(
            frame, (lshoulder_x, lshoulder_y), (lelbow_x, lelbow_y), (0, 255, 0), 5
        )
        cv2.line(frame, (lelbow_x, lelbow_y), (lwrist_x, lwrist_y), (0, 255, 0), 5)
        cv2.circle(frame, (lshoulder_x, lshoulder_y), 5, (255, 0, 0), -1)
        cv2.circle(frame, (lelbow_x, lelbow_y), 5, (255, 0, 0), -1)
        cv2.circle(frame, (lwrist_x, lwrist_y), 5, (255, 0, 0), -1)
        # Draw Left lines on the frame
        cv2.rectangle(frame, (395, 430), (640, 480), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"Lt angle --{str(langle)}",
            (400, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        # Draw RIGHT angle on the frame
        cv2.putText(
            frame,
            f"Rt angle --{str(rangle)}",
            (400, 470),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.line(
            frame, (rshoulder_x, rshoulder_y), (relbow_x, relbow_y), (0, 255, 0), 5
        )
        cv2.line(frame, (relbow_x, relbow_y), (rwrist_x, rwrist_y), (0, 255, 0), 5)
        cv2.circle(frame, (rshoulder_x, rshoulder_y), 5, (255, 0, 0), -1)
        cv2.circle(frame, (relbow_x, relbow_y), 5, (255, 0, 0), -1)
        cv2.circle(frame, (rwrist_x, rwrist_y), 5, (255, 0, 0), -1)

    return [frame, rangle, langle]


def start(frame,x,y):
    cv2.rectangle(frame, (310, 0), (390, 35), (0, 0, 0), -1)
    cv2.putText(
            frame,
            "Start",
            (320, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
    )
    if x > 310 and x < 350 and y > 10 and y < 55:
        return 1
    return 0


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)


def main():
    python_interpreter = f"{sys.executable}"
    call_script = None
    cnt = 0
    stage = "down"
    w, h = 640, 480
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    flag=0
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        result = mp_hands.process(frame_rgb)
        x, y = -1, -1
        if result.multi_hand_landmarks:
            # Get the first (right) hand's landmarks
            hand_landmarks = result.multi_hand_landmarks[0]

            # Get the right index finger coordinates (landmark 8)
            index_finger_landmark = hand_landmarks.landmark[
                mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP
            ]
            x, y = int(index_finger_landmark.x * w), int(index_finger_landmark.y * h)
        cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
        if not flag:
            flag=start(frame,x,y)
        # Process the frame and draw lines
        if flag==1:
            try:
                output_frame, rangle, langle = shoulderpressutils(frame)
                if rangle > 140 and langle > 140:
                    stage = "down"
                if rangle < 70 and langle < 70 and stage == "down":
                    stage = "up"
                    cnt += 1
                cv2.rectangle(frame, (0, 0), (60, 80), (0, 0, 0), -1)
                cv2.putText(
                    output_frame,
                    str(cnt),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            except:
                output_frame = frame
        if not flag:
            output_frame = frame
        cv2.rectangle(frame, (505, 10), (640, 55), (0, 0, 0), -1)
        cv2.putText(
            output_frame,
            "BACK",
            (515, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if x > 505 and x < 640 and y > 10 and y < 55:
            cv2.putText(
                frame,
                "BACK",
                (425, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            call_script = "Menu.py"
        command = [python_interpreter, call_script]
        if call_script is not None:
            try:
                subprocess.Popen(command)
                break
            except subprocess.CalledProcessError as e:
                print(f"Error executing the script: {e}")
            except FileNotFoundError:
                print("Enter the correct PATH.")
        cv2.imshow("Output", output_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
