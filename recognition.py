import face_recognition
import os, sys
import cv2
import numpy as np
import math
import mediapipe as mp
import time
import subprocess
import datetime
import random

class UserAttributes:
    def __init__(self):
        self.user = "None"
        self.cnt1 = 0
        self.cnt2 = 0
        self.cnt3 = 0
        self.cnt4 = 0
        self.cnt5 = 0
        self.cnt6 = 0

    def username(self, name):
        self.user = name

    def barbellcnt(self, cnt):
        self.cnt1 = cnt

    def bicepcnt(self, cnt):
        self.cnt2 = cnt

    def frntraisecnt(self, cnt):
        self.cnt3 = cnt

    def jjackscnt(self, cnt):
        self.cnt4 = cnt

    def shldrcnt(self, cnt):
        self.cnt5 = cnt

    def sqcnt(self, cnt):
        self.cnt6 = cnt

    def saveall(self):
        arr = [self.cnt1, self.cnt2, self.cnt3, self.cnt4, self.cnt5, self.cnt6]
        arr = np.array(arr)
        print(arr)
        with open("counts.txt", "a+") as file:
            today = datetime.date.today()
            date_str = today.strftime("%Y-%m-%d")
            file.seek(0)
            lines = file.readlines()
            file.write(f"{self.user}: {date_str} --> {arr[0]} - {arr[1]} - {arr[2]} - {arr[3]} - {arr[4]} - {arr[5]}\n")
            file.seek(0)
            file.writelines(lines)


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# Helper
def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


def ask(x,y):
        if x > 10 and x < 120 and y > 100 and y < 135:
            return 1
        return 0

class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f"faces/{image}")
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
    
    
    def encode_realtime(self,path,name):
        face_image = face_recognition.load_image_file(path)
        face_encoding = face_recognition.face_encodings(face_image)[0]

        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(name)


    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)
        flag=0
        w, h = 640, 480
        f2=0
        namee1="Unknown"
        mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        while True:
            ret, frame = video_capture.read()
            rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            result = mp_hands.process(rgb_frame)
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
            # Only process every other frame of video to save time
            if self.process_current_frame:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

                # Find all the faces and face encodings in the current frame of video
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = '???'

                    # Calculate the shortest distance to face
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        namee1=name
                        confidence = face_confidence(face_distances[best_match_index])
                        self.name=namee1
                    self.face_names.append(f'{name} ({confidence})')
                    if not matches[best_match_index]:
                        f2=1
            if f2==1:
                flag=ask(x,y)
                if flag==1:
                    pathh,namee=fr.register(frame)
                    fr.encode_realtime(pathh,namee) 
            
            if namee1 != "Unknown":
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
                            command = [f"{sys.executable}", "Registered.py"]
                            subprocess.Popen(command)
                            break
            
            else:
                 cv2.rectangle(frame, (10, 100), (120, 135), (0, 0, 0), -1)
                 cv2.putText(
                                frame,
                                "Register",
                                 (15, 115),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )
            self.process_current_frame = not self.process_current_frame

            # Display the results
            check=""
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Create the frame with the name
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Face Registration', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) == ord('q'):
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()
    
    
    def register(self,frame):
        #insert name entered from the app
        image_filename = f"{random.randint(0,100)}.png"
        image_path = os.path.join("faces", image_filename)
        cv2.imwrite(image_path, frame)
        print(f"Image saved at: {image_path}")
        return image_path,image_filename


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()