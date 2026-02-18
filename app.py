import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import face_recognition
import numpy as np
import av
import pandas as pd
from datetime import datetime
from database_manager import StudentDB

# --- ตั้งค่า RTC สำหรับใช้งานผ่าน Internet (STUN Server) ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="Smart Reg Mobile", layout="wide")
db = StudentDB()

st.title("📱 ระบบลงทะเบียนอัจฉริยะ (Mobile Web)")

# --- ฟังก์ชันประมวลผลวิดีโอ (หัวใจหลักของ Mobile App) ---
class VideoProcessor:
    def __init__(self):
        self.db = db

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # ลดขนาดภาพเพื่อเพิ่มความเร็วบนมือถือ
        small_img = np.ascontiguousarray(img[::2, ::2, ::-1])
        
        face_locations = face_recognition.face_locations(small_img)
        face_encodings = face_recognition.face_encodings(small_img, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.db.data["encodings"], face_encoding, tolerance=0.5)
            name = "Unknown"
            color = (0, 0, 255) # แดง ถ้าไม่รู้จัก

            if True in matches:
                first_match_index = matches.index(True)
                name = self.db.data["names"][first_match_index]
                color = (0, 255, 0) # เขียว ถ้าพบชื่อ

            # วาดกรอบบนหน้าจอมือถือ
            top *= 2; right *= 2; bottom *= 2; left *= 2
            cv2.rectangle(img, (left, top), (right, bottom), color, 2)
            cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI ส่วนการสแกน ---
st.subheader("🔍 สแกนใบหน้าผ่านมือถือ")
webrtc_ctx = webrtc_streamer(
    key="face-reg",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# --- ส่วนการยืนยันตัวตนด้วยเสียง (สไตล์ Mobile UX) ---
if webrtc_ctx.state.playing:
    st.write("---")
    st.info("🎯 หากระบบพบใบหน้าของคุณแล้ว กดปุ่มไมค์ด้านล่างเพื่อพูดชื่อ")
    if st.button("🎤 ยืนยันตัวตนด้วยเสียง"):
        # เรียกใช้ฟังก์ชันรับเสียงที่เราทำไว้ก่อนหน้า
        # (หมายเหตุ: SpeechRecognition บน Browser มือถือต้องระวังเรื่อง Permission)
        pass

# --- ส่วนรายงานสถิติวันนี้ ---
st.write("---")
st.subheader("📊 ยอดผู้มาเรียนวันนี้")
if os.path.exists("attendance_log.csv"):
    df = pd.read_csv("attendance_log.csv")
    st.dataframe(df.tail(5), use_container_width=True)
