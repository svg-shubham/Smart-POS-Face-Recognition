import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox
import cv2
import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
import csv

# Theme Settings
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ModernAttendancePOS:
    def __init__(self, root):
        # 1. ROOT INITIALIZATION (Sabse pehle define karein)
        self.root = root
        self.root.title("Kalyani Collections Islapur")
        self.root.geometry("1200x800")

        # 2. WINDOW ICON SETTING (Taskbar/Title Bar Icon)
        try:
            icon_path = "download.png"
            if os.path.exists(icon_path):
                img = Image.open(icon_path)
                self.app_icon = ImageTk.PhotoImage(img) # Save in self to avoid garbage collection
                self.root.iconphoto(False, self.app_icon)
        except Exception as e:
            print(f"Icon Load Error: {e}")

        # 3. DATA FOLDERS INITIALIZATION
        for folder in ['data', 'Attendance']:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # 4. VARIABLES
        self.cap = None
        self.is_running = False
        self.record = None
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        # --- SIDEBAR (McDonald's POS Style) ---
        self.sidebar = ctk.CTkFrame(self.root, width=280, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")

        # Sidebar Logo Integration
        try:
            logo_path = "download.png"
            if os.path.exists(logo_path):
                logo_img = Image.open(logo_path)
                self.logo_image = ctk.CTkImage(light_image=logo_img, dark_image=logo_img, size=(120, 120))
                self.logo_label_img = ctk.CTkLabel(self.sidebar, image=self.logo_image, text="")
                self.logo_label_img.pack(pady=(40, 10))
        except:
            print("Sidebar logo not found.")

        self.logo_label = ctk.CTkLabel(self.sidebar, text="KALYANI COLLECTIONS", 
                                       font=ctk.CTkFont(size=18, weight="bold"), text_color="#1abc9c")
        self.logo_label.pack(pady=(0, 40))

        # Navigation Buttons
        self.btn_add = self.create_nav_btn("ADD NEW PROFILE", "#e67e22", "#d35400", self.ui_add_profile)
        self.btn_att = self.create_nav_btn("MARK ATTENDANCE", "#27ae60", "#219150", self.ui_take_attendance)
        self.btn_rec = self.create_nav_btn("VIEW LOGS / CSV", "#2980b9", "#2471a3", self.ui_view_records)
        
        self.btn_exit = ctk.CTkButton(self.sidebar, text="SHUTDOWN", fg_color="transparent", 
                                      border_width=1, border_color="#c0392b", text_color="#c0392b", 
                                      command=self.root.quit)
        self.btn_exit.pack(side="bottom", fill="x", padx=20, pady=20)

        # --- MAIN DISPLAY CONTAINER ---
        self.main_container = ctk.CTkFrame(self.root, corner_radius=15, fg_color="#121212")
        self.main_container.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        self.show_welcome()

    def create_nav_btn(self, text, color, hover, cmd):
        btn = ctk.CTkButton(self.sidebar, text=text, height=60, corner_radius=12, 
                             fg_color=color, hover_color=hover, 
                             font=ctk.CTkFont(size=14, weight="bold"), command=cmd)
        btn.pack(fill="x", padx=20, pady=12)
        return btn

    def clear_main(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        for widget in self.main_container.winfo_children():
            widget.destroy()

    def show_welcome(self):
        self.clear_main()
        ctk.CTkLabel(self.main_container, text="SYSTEM READY\nKalyani Collections Islapur", 
                     font=ctk.CTkFont(size=30, weight="bold"), text_color="#2c3e50").pack(expand=True)

    # --- OPERATION 1: ADD PROFILE ---
    def ui_add_profile(self):
        self.clear_main()
        self.face_data = []

        ctk.CTkLabel(self.main_container, text="USER REGISTRATION", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=20)

        card = ctk.CTkFrame(self.main_container, fg_color="#1e1e1e", corner_radius=15)
        card.pack(pady=10, padx=40, fill="x")

        self.name_entry = ctk.CTkEntry(card, placeholder_text="Enter Full Name", width=400, height=50, font=ctk.CTkFont(size=16))
        self.name_entry.pack(side="left", padx=20, pady=20)

        self.btn_start = ctk.CTkButton(card, text="START SCAN", width=180, height=50, fg_color="#e67e22", command=self.start_capture_loop)
        self.btn_start.pack(side="left", padx=10)

        self.video_frame = tk.Label(self.main_container, bg="#121212")
        self.video_frame.pack(pady=20)

        self.p_label = ctk.CTkLabel(self.main_container, text="Progress: 0%", font=ctk.CTkFont(size=16))
        self.p_label.pack()
        self.p_bar = ctk.CTkProgressBar(self.main_container, width=600, height=20, progress_color="#e67e22")
        self.p_bar.set(0)
        self.p_bar.pack(pady=10)

    def start_capture_loop(self):
        if not self.name_entry.get().strip():
            messagebox.showwarning("Input Error", "Please enter a name first!")
            return
        self.is_running = True
        self.cap = cv2.VideoCapture(0)
        self.capture_step()

    def capture_step(self):
        if self.is_running:
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (230, 126, 34), 3)
                    if len(self.face_data) < 100:
                        crop = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                        self.face_data.append(crop)
                        self.p_bar.set(len(self.face_data) / 100)
                        self.p_label.configure(text=f"Scanning: {len(self.face_data)}/100")

                self.display_video(frame)
                if len(self.face_data) >= 100:
                    self.save_profile()
                    return
                self.root.after(10, self.capture_step)

    def save_profile(self):
        name = self.name_entry.get().strip()
        data = np.asarray(self.face_data).reshape(100, -1)
        
        n_path = 'data/names.pkl'
        names = [name]*100
        if os.path.exists(n_path):
            with open(n_path, 'rb') as f: names = pickle.load(f) + names
        with open(n_path, 'wb') as f: pickle.dump(names, f)

        f_path = 'data/faces_data.pkl'
        if os.path.exists(f_path):
            with open(f_path, 'rb') as f: data = np.append(pickle.load(f), data, axis=0)
        with open(f_path, 'wb') as f: pickle.dump(data, f)

        self.clear_main()
        messagebox.showinfo("Success", f"Profile created for {name}")
        self.show_welcome()

    # --- OPERATION 2: MARK ATTENDANCE ---
    def ui_take_attendance(self):
        self.clear_main()
        if not os.path.exists('data/names.pkl'):
            messagebox.showerror("Error", "No registered profiles found!")
            return

        with open('data/names.pkl', 'rb') as f: labels = pickle.load(f)
        with open('data/faces_data.pkl', 'rb') as f: faces = pickle.load(f)
        self.knn = KNeighborsClassifier(n_neighbors=5)
        self.knn.fit(faces, labels)

        ctk.CTkLabel(self.main_container, text="POS FACE SCANNER", font=ctk.CTkFont(size=24, weight="bold"), text_color="#27ae60").pack(pady=20)
        
        self.video_frame = tk.Label(self.main_container, bg="#121212")
        self.video_frame.pack()

        self.btn_confirm = ctk.CTkButton(self.main_container, text="CONFIRM & LOG ATTENDANCE", 
                                          height=70, width=500, fg_color="#27ae60", 
                                          hover_color="#219150", font=ctk.CTkFont(size=18, weight="bold"), 
                                          command=self.log_attendance)
        self.btn_confirm.pack(pady=30)

        self.is_running = True
        self.cap = cv2.VideoCapture(0)
        self.record = None
        self.recognize_step()

    def recognize_step(self):
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    crop = cv2.resize(frame[y:y+h, x:x+w], (50, 50)).flatten().reshape(1, -1)
                    detected_name = self.knn.predict(crop)[0]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (39, 174, 96), 3)
                    cv2.putText(frame, str(detected_name), (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    self.record = [str(detected_name), datetime.now().strftime("%H:%M:%S")]

                self.display_video(frame)
                self.root.after(10, self.recognize_step)

    def log_attendance(self):
        if self.record:
            date = datetime.now().strftime("%d-%m-%Y")
            path = f"Attendance/Attendance_{date}.csv"
            exists = os.path.isfile(path)
            with open(path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not exists: writer.writerow(["NAME", "TIME"])
                writer.writerow(self.record)
            messagebox.showinfo("POS SUCCESS", f"Attendance Logged:\n{self.record[0]}")
        else:
            messagebox.showwarning("Warning", "No face detected. Please face the camera.")

    # --- OPERATION 3: VIEW RECORDS ---
    def ui_view_records(self):
        self.clear_main()
        ctk.CTkLabel(self.main_container, text="ATTENDANCE RECORDS", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=20)
        
        date = datetime.now().strftime("%d-%m-%Y")
        path = f"Attendance/Attendance_{date}.csv"

        if os.path.exists(path):
            df = pd.read_csv(path)
            table_box = ctk.CTkTextbox(self.main_container, width=800, height=450, font=("Consolas", 14))
            table_box.pack(pady=10)
            table_box.insert("0.0", df.to_string(index=False))
            
            ctk.CTkButton(self.main_container, text="OPEN FOLDER", width=300, command=lambda: os.startfile(os.path.abspath("Attendance"))).pack(pady=20)
        else:
            ctk.CTkLabel(self.main_container, text="No records found for today.", text_color="#c0392b").pack(pady=50)

    def display_video(self, frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_frame.imgtk = imgtk
        self.video_frame.configure(image=imgtk)

if __name__ == "__main__":
    root = ctk.CTk()
    app = ModernAttendancePOS(root)
    root.mainloop()