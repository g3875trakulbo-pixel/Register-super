import face_recognition
import pandas as pd
import os
import pickle
from datetime import datetime

class StudentDB:
    def __init__(self, db_path="face_db.pkl", log_path="attendance_log.csv"):
        self.db_path = db_path
        self.log_path = log_path
        self.data = self.load_db()

    def load_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                return pickle.load(f)
        return {"encodings": [], "names": [], "ids": []}

    def save_student(self, face_image, name, student_id):
        # ขั้นตอนที่ 3: จัดเก็บลงฐานข้อมูล
        encodings = face_recognition.face_encodings(face_image)
        if encodings:
            self.data["encodings"].append(encodings[0])
            self.data["names"].append(name)
            self.data["ids"].append(student_id)
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.data, f)
            return True
        return False

    def log_attendance(self, name, voice_text):
        # ขั้นตอนที่ 7: บันทึกข้อความเสียงและเวลาลง CSV
        now = datetime.now()
        new_entry = {
            "ชื่อ-นามสกุล": name,
            "ข้อความยืนยัน": voice_text,
            "วันที่": now.strftime("%Y-%m-%d"),
            "เวลา": now.strftime("%H:%M:%S")
        }
        df = pd.DataFrame([new_entry])
        df.to_csv(self.log_path, mode='a', header=not os.path.exists(self.log_path), index=False, encoding='utf-8-sig')
