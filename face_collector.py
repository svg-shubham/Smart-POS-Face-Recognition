import cv2
import pickle
import numpy as np
import os

def collect_data(name):
    if not os.path.exists('data'): os.makedirs('data')
    
    video = cv2.VideoCapture(0)
    faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces_data = []
    i = 0

    while True:
        ret, frame = video.read()
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(grey, 1.3, 5)
        
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50))
            if len(faces_data) < 100 and i % 10 == 0:
                faces_data.append(resized_img)
            i += 1
            cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord('q') or len(faces_data) == 100:
            break

    video.release()
    cv2.destroyAllWindows()

    faces_data = np.asarray(faces_data).reshape(len(faces_data), -1)

    # Save Names
    names_path = 'data/names.pkl'
    if not os.path.exists(names_path):
        names = [name] * len(faces_data)
    else:
        with open(names_path, 'rb') as f: names = pickle.load(f)
        names += [name] * len(faces_data)
    with open(names_path, 'wb') as f: pickle.dump(names, f)

    # Save Faces
    faces_path = 'data/faces_data.pkl'
    if not os.path.exists(faces_path):
        with open(faces_path, 'wb') as f: pickle.dump(faces_data, f)
    else:
        with open(faces_path, 'rb') as f: existing_faces = pickle.load(f)
        all_faces = np.append(existing_faces, faces_data, axis=0)
        with open(faces_path, 'wb') as f: pickle.dump(all_faces, f)