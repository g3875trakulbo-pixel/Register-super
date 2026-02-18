import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import mediapipe as mp
import cv2
import av

# --- เรียกใช้ Mediapipe แบบปลอดภัยที่สุด ---
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# ค่า STUN เพื่อให้รันผ่านเน็ตมือถือได้
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
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                
                # วาดกรอบสแกน
                cv2.rectangle(img, bbox, (0, 255, 0), 2)
                cv2.putText(img, "SCANNING...", (bbox[0], bbox[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.info("ขั้นตอน: 1.กดปุ่ม Start 2.อนุญาตให้ใช้กล้อง")

webrtc_streamer(
    key="mobile-scan",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if st.button("📝 ลงทะเบียน"):
    st.balloons()
    st.success("บันทึกการลงทะเบียนเบื้องต้นสำเร็จ!")
