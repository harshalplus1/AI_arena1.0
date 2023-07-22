import cv2
import subprocess
import time
import sys
import mediapipe as mp


cap=cv2.VideoCapture(0)
mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
w,h=640,480
while True:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x, y = -1, -1
    result = mp_hands.process(frame_rgb)
    if result.multi_hand_landmarks:
            # Get the first (right) hand's landmarks
        hand_landmarks = result.multi_hand_landmarks[0]

            # Get the right index finger coordinates (landmark 8)
        index_finger_landmark = hand_landmarks.landmark[
            mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP
            ]
        x, y = int(index_finger_landmark.x * w), int(index_finger_landmark.y * h)
    cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
    cv2.rectangle(frame, (240, 0), (470, 35), (0, 0, 0), -1)
    cv2.putText(frame, "Registered Sucessfully!", (250, 15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )
    cv2.putText(frame, "Point here to Continue", (250, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )
    cv2.imshow('Face Registration', frame)
    if x > 310 and x < 350 and y > 10 and y < 55:
        command = [f"{sys.executable}", "Menu.py"]
        subprocess.Popen(command)
        break
        
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()