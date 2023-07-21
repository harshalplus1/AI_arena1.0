import cv2
import subprocess
import time
import sys

cap=cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    cv2.rectangle(frame, (240, 0), (470, 35), (0, 0, 0), -1)
    cv2.putText(frame, "Registered Sucessfully!", (250, 15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )
    cv2.putText(frame, "Redirecting to the app", (250, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )
    cv2.imshow('Face Registration', frame)  
    time.sleep(5)
    command = [f"{sys.executable}", "Menu.py"]
    subprocess.Popen(command)
    break
        
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()