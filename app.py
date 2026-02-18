import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import mediapipe as mp
import cv2
import av

# เรียกใช้โมดูลแบบเจาะจงเพื่อเลี่ยง Error
mp_drawing = mp.solutions.drawing_utils
mp_face_det = mp.solutions.face_detection
face_detection = mp_face_det.FaceDetection(model_selection=0, min_detection_confidence=0.5)

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.title("🤖 Smart Reg Takad School")

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(img_rgb)

        if results.detections:
            for detection in results.detections:
                # วาดกรอบสแกนด้วยเครื่องมือมาตรฐานของ Mediapipe
                mp_drawing.draw_detection(img, detection)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.info("ขั้นตอน: 1.กดปุ่ม START 2.อนุญาตให้ใช้กล้อง")

webrtc_streamer(
    key="mobile-scan-v1",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if st.button("📝 ยืนยันการมาเรียน"):
    st.balloons()
    st.success("บันทึกข้อมูลเรียบร้อย!")
