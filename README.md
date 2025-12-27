Smart POS Face Recognition System
A Modern, AI-Powered Attendance & User Management Solution

📌 Overview
This project is a Modern Point-of-Sale (POS) style Attendance System designed for retail environments like Kalyani Collections. Unlike traditional attendance systems, it uses Computer Vision and Machine Learning (KNN) to identify employees/users in real-time, providing a seamless and touchless experience.

The UI is inspired by high-end kiosks (like McDonald's POS) to ensure a high-contrast, touch-friendly, and professional aesthetic.

✨ Key Features
Modern UI/UX: Built with CustomTkinter for a dark-themed, sleek, and responsive interface.

Real-time Recognition: Uses OpenCV and Haar Cascades for lightning-fast face detection.

Machine Learning Integration: Implements a K-Nearest Neighbors (KNN) classifier for high-accuracy face identification.

Branded Experience: Integrated business branding (Logo, Window Icons, and Business Name).

Automated Logging: Saves attendance records directly into daily-organized CSV files for easy HR processing.

Secure Data Handling: Uses Pickle for local face-data storage.

🛠️ Tech Stack
Language: Python

Computer Vision: OpenCV

Machine Learning: Scikit-Learn (KNN)

GUI Library: CustomTkinter (Modern Tkinter wrapper)

Data Handling: Pandas, NumPy, Pickle, CSV

📸 Screenshots
(please use your own screenshots and file for that)

🚀 Installation & Setup
Clone the repository:

Bash

git clone https://github.com/YOUR_USERNAME/Smart-POS-Face-Recognition.git
cd Smart-POS-Face-Recognition
Install Dependencies:

Bash

pip install -r requirements.txt
Run the Application:

Bash

python main.py
📂 Project Structure
Plaintext

├── data/               # Stores Pickle files for names and face data
├── Attendance/         # Daily attendance CSV logs generated here
├── main.py             # Main Application Logic
├── download.png        # Business Logo
└── requirements.txt    # List of dependencies
🤝 Contributing
Contributions are welcome! If you have any ideas to improve the UI or the recognition algorithm, feel free to fork this repo and submit a PR.
