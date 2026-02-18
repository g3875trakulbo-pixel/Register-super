import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import mediapipe as mp
import cv2
import numpy as np
import av
import pandas as pd
from datetime import datetime

# --- ตั้งค่า Mediapipe ให้ถูกต้อง ---
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.title("🤖 Smart Registration Mobile")
st.write("โรงเรียนตากาดประชาสามัคคี")

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # ตรวจจับใบหน้า
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(img_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                
                # วาดกรอบ
                cv2.rectangle(img, bbox, (0, 255, 0), 2)
                cv2.putText(img, "Face Detected", (bbox[0], bbox[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- ส่วนควบคุมหน้าเว็บ ---
st.info("ขั้นตอน: 1.กดปุ่ม Start ด้านล่าง 2.อนุญาตให้ใช้กล้อง")

webrtc_streamer(
    key="mobile-scan",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if st.button("🎤 ลงชื่อด้วยเสียง (ทดสอบ)"):
    st.balloons()
    st.success("บันทึกข้อมูลการลงทะเบียนเบื้องต้นสำเร็จ!")
