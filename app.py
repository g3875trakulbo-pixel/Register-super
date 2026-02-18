import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import mediapipe as mp
import cv2
import numpy as np
import av
import pandas as pd
from datetime import datetime

# --- การตั้งค่า AI ตรวจจับใบหน้า ---
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.title("🤖 Smart Reg Mobile (Takad School)")

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # ใช้ Mediapipe ตรวจจับใบหน้า (เบาและเร็ว)
        results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                
                # วาดกรอบสีเขียวรอบใบหน้า
                cv2.rectangle(img, bbox, (0, 255, 0), 2)
                cv2.putText(img, "Face Detected", (bbox[0], bbox[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- ส่วนแสดงผลบนมือถือ ---
st.write("ขั้นตอน: 1.เปิดกล้อง 2.ยืนยันตัวตน")
webrtc_streamer(
    key="mobile-reg",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if st.button("🎤 พูดชื่อเพื่อลงทะเบียน"):
    st.success("กำลังพัฒนาระบบรับเสียงบน Cloud... (ทดสอบบันทึกสำเร็จ)")
