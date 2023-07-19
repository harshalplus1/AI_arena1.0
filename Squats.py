#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import mediapipe as mp
from Angle import calculate_angle
import subprocess
import sys


# In[2]:


mp_drawing = mp.solutions.drawing_utils


# In[3]:


def squatsutil(frame):
    # Convert frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Pose model
    results = pose.process(frame_rgb)
    # Check if any poses are detected
    if results.pose_landmarks:
        # Get landmark coordinates
        landmarks = results.pose_landmarks.landmark
        # results = holistic.process(frame_rgb)
        w, h = 640, 480
        # Specify LEFT shoulder, elbow, and hip landmark indices
        l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        lank = [
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
        ]
        lknee = [
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
        ]
        lhip = [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
        ]
        # Specify RIGHT shoulder, elbow, and hip landmark indices
        r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        rank = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
        ]
        rknee = [
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
        ]
        rhip = [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
        ]
        # Convert LEFT landmark positions to pixel coordinates
        l_ankle_x, l_ankle_y = int(l_ankle.x * w), int(l_ankle.y * h)
        l_knee_x, l_knee_y = int(l_knee.x * w), int(l_knee.y * h)
        l_hip_x, l_hip_y = int(l_hip.x * w), int(l_hip.y * h)
        langle = calculate_angle(lhip, lknee, lank)
        # Convert RIGHT landmark positions to pixel coordinates
        r_ankle_x, r_ankle_y = int(r_ankle.x * w), int(r_ankle.y * h)
        r_knee_x, r_knee_y = int(r_knee.x * w), int(r_knee.y * h)
        r_hip_x, r_hip_y = int(r_hip.x * w), int(r_hip.y * h)
        rangle = calculate_angle(rhip, rknee, rank)
        cv2.line(frame, (l_ankle_x, l_ankle_y), (l_knee_x, l_knee_y), (0, 255, 0), 5)
        cv2.line(frame, (l_knee_x, l_knee_y), (l_hip_x, l_hip_y), (0, 255, 0), 5)
        cv2.circle(frame, (l_ankle_x, l_ankle_y), 5, (255, 0, 0), -1)
        cv2.circle(frame, (l_knee_x, l_knee_y), 5, (255, 0, 0), -1)
        cv2.circle(frame, (l_hip_x, l_hip_y), 5, (255, 0, 0), -1)
        # right
        cv2.line(frame, (r_ankle_x, r_ankle_y), (r_knee_x, r_knee_y), (0, 255, 0), 5)
        cv2.line(frame, (r_knee_x, r_knee_y), (r_hip_x, r_hip_y), (0, 255, 0), 5)
        cv2.circle(frame, (r_ankle_x, r_ankle_y), 5, (255, 0, 0), -1)
        cv2.circle(frame, (r_knee_x, r_knee_y), 5, (255, 0, 0), -1)
        cv2.circle(frame, (r_hip_x, r_hip_y), 5, (255, 0, 0), -1)
    return [frame, rangle, langle]


# In[4]:


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
)
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

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
    rangle, langle=0,0
    while True:
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                output_frame, rangle, langle = squatsutil(frame)
                if rangle > 120 and langle > 120:
                    stage = "up"
                if rangle < 70 and langle < 70 and stage == "up":
                    stage = "down"
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
                cv2.rectangle(frame, (395, 430), (640, 480), (0, 0, 0), -1)
                # Draw LEFT angle on the frame
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
            except:
                output_frame = frame
        if not flag:
            output_frame=frame
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
