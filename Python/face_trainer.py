import cv2
import os
import numpy as np
import pickle

CASCADE_MODEL_DIR = "../Models/haarcascade_frontalface_default.xml"
EMBEDDED_MODEL_DIR = "../Models/face_recognizer.yaml"
PICKLE_FILE_DIR = "../Models/labels.pickle"
VIDEOS_DATA_DIR = "../Faces"  # Replace with your folder containing video files

cascade = cv2.CascadeClassifier(CASCADE_MODEL_DIR)
recognise = cv2.face.LBPHFaceRecognizer_create()


def getdata():
    current_id = 0
    label_id = {}
    face_train = []
    face_label = []

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    my_video_dir = os.path.join(BASE_DIR, VIDEOS_DATA_DIR)

    for root, dirs, files in os.walk(my_video_dir):
        for file in files:
            if file.lower().endswith(("mp4", "avi", "mov", "webm")):
                path = os.path.join(root, file)
                label = os.path.basename(root).lower()

                if label not in label_id:
                    label_id[label] = current_id
                    current_id += 1
                ID = label_id[label]

                # Open the video file
                cap = cv2.VideoCapture(path)

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5
                    )

                    for x, y, w, h in faces:
                        roi = gray[y : y + h, x : x + w]
                        cv2.imshow("Training on Face", roi)
                        cv2.waitKey(1)
                        face_train.append(roi)
                        face_label.append(ID)

                cap.release()

    cv2.destroyAllWindows()

    with open(PICKLE_FILE_DIR, "wb") as f:
        pickle.dump(label_id, f)

    return face_train, face_label


# Training and saving the model
faces, ids = getdata()
recognise.train(faces, np.array(ids))
recognise.save(EMBEDDED_MODEL_DIR)
