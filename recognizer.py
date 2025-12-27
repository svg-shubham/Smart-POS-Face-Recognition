import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier

def start_recognition():
    if not os.path.exists('data/names.pkl'): return "No Data Found"
    
    with open('data/names.pkl', 'rb') as f: LABELS = pickle.load(f)
    with open('data/faces_data.pkl', 'rb') as f: FACES = pickle.load(f)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)

    video = cv2.VideoCapture(0)
    faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = video.read()
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(grey, 1.3, 5)
        
        attendance_info = None
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)
            
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            attendance_info = [str(output[0]), str(timestamp), date]

        cv2.imshow("Attendance (Press 'o' to Log, 'q' to Exit)", frame)
        k = cv2.waitKey(1)
        if k == ord('o') and attendance_info:
            file_path = f"Attendance/Attendance_{attendance_info[2]}.csv"
            if not os.path.exists("Attendance"): os.makedirs("Attendance")
            file_exists = os.path.isfile(file_path)
            with open(file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists: writer.writerow(["NAME", "TIME"])
                writer.writerow([attendance_info[0], attendance_info[1]])
            print("Attendance Logged!")
        
        if k == ord('q'): break

    video.release()
    cv2.destroyAllWindows()