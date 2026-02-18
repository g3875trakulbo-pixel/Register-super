import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import mediapipe as mp
import cv2
import numpy as np
import av

# --- โครงสร้างที่ปลอดภัยที่สุดในการเรียกใช้ Mediapipe ---
mp_face_module = mp.solutions.face_detection
face_detector = mp_face_module.FaceDetection(min_detection_confidence=0.5)

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.title("🤖 Smart Reg Mobile (Takad School)")

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # แปลงสีภาพ
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # ตรวจจับใบหน้า
        results = face_detector.process(img_rgb)

        if results.detections:
            for detection in results.detections:
                # วาดกรอบเบื้องต้น
                st.session_state['face_detected'] = True
                ih, iw, _ = img.shape
                bboxC = detection.location_data.relative_bounding_box
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(img, bbox, (0, 255, 0), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.info("ขั้นตอน: 1.กดปุ่ม Start 2.เมื่อเห็นกรอบเขียว ให้กดปุ่มลงทะเบียน")

webrtc_streamer(
    key="registration-mobile",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

if st.button("📝 ยืนยันการลงทะเบียน"):
    st.balloons()
    st.success("บันทึกข้อมูลสำเร็จ! (จำลองสถานะมาเรียน)")
