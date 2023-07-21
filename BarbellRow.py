#!/usr/bin/env python
# coding: utf-8



import numpy as np
import cv2
import mediapipe as mp
from Angle import calculate_angle
import subprocess
import sys

mp_drawing = mp.solutions.drawing_utils


def barbellrowutil(frame):
    # Convert frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Pose model
    results = pose.process(frame_rgb)

    # Check if any poses are detected
    if results.pose_landmarks:
        # Get landmark coordinates
        landmarks = results.pose_landmarks.landmark
        results = holistic.process(frame_rgb)
        w, h = 640, 480
        # Specify LEFT shoulder, elbow, and wrist landmark indices
        l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        l_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
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
        lhip = [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
        ]
        lvert = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 0]


        # Convert LEFT landmark positions to pixel coordinates
        lshoulder_x, lshoulder_y = int(l_shoulder.x * w), int(l_shoulder.y * h)
        lelbow_x, lelbow_y = int(l_elbow.x * w), int(l_elbow.y * h)
        lwrist_x, lwrist_y = int(l_wrist.x * w), int(l_wrist.y * h)
        l_hip_x, l_hip_y = int(l_hip.x * w), int(l_hip.y * h)
        lvert_x, lvert_y = int(l_shoulder.x * w), int(0 * h)
        langle = calculate_angle(lsh, lel, lwr)
        posture_angle = calculate_angle(lsh, lhip, lvert)

        # Draw LEFT lines on the frame
        cv2.rectangle(frame, (395, 430), (640, 460), (0, 0, 0), -1)
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
        cv2.line(
            frame, (lshoulder_x, lshoulder_y), (lelbow_x, lelbow_y), (0, 255, 0), 5
        )
        cv2.line(frame, (lelbow_x, lelbow_y), (lwrist_x, lwrist_y), (0, 255, 0), 5)
        if posture_angle > 10 and posture_angle < 30:
            cv2.line(
                frame, (lshoulder_x, lshoulder_y), (l_hip_x, l_hip_y), (0, 255, 0), 5
            )
        else:
            cv2.line(
                frame, (lshoulder_x, lshoulder_y), (l_hip_x, l_hip_y), (0, 0, 255), 5
            )
            cv2.putText(
                frame,
                "BAD POSTURE",
                (220, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.circle(frame, (lshoulder_x, lshoulder_y), 5, (255, 0, 0), -1)
        cv2.circle(frame, (lelbow_x, lelbow_y), 5, (255, 0, 0), -1)
        cv2.circle(frame, (lwrist_x, lwrist_y), 5, (255, 0, 0), -1)
        cv2.circle(frame, (l_hip_x, l_hip_y), 5, (255, 0, 0), -1)
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=4),
        )
    return [frame, langle, posture_angle]



mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
)
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
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
    # Change this if your Python executable has a different name or path
    call_script = None
    cnt = 0
    stage = "down"
    w, h = 640, 480
    flag=0
    cap = cv2.VideoCapture(0)
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
                output_frame, langle, posture_angle = barbellrowutil(frame)
                if langle > 150:
                    stage = "back"
                if langle < 108 and stage == "back":
                    stage = "front"
                    cnt += 1
                cv2.rectangle(frame, (0, 0), (90, 80), (0, 0, 0), -1)
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
                try:
                    with open('output.txt', 'a+') as file:
                        file.write(f"BarbellRow: '{cnt}'" + "\n")
                except Exception as e:
                    print(f"error : {e}")
                subprocess.Popen(command)
                break
            except subprocess.CalledProcessError as e:
                print(f"Error executing the script: {e}")
            except FileNotFoundError:
                print("Enter the correct PATH.")
        cv2.imshow("BarbellRow", output_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
