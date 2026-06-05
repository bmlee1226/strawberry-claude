import cv2
import streamlit as st

from src.disease_data import disease_info
from src.data_models import DetectionResult, VideoInfo

def show_disease_info(class_id: int) -> None:

    info = disease_info.get(class_id)

    st.header(f"🩺 {info['name']}  병해충 정보")

    with st.container(border=True):

        st.write(info["symptom"])
        st.write(info["cause"])
        st.write(info["solution"])

        st.write("🍓 병해 예시 이미지")

        st.image(info["image"])

        st.caption(info["name"])


def parse_detection_result(results) -> DetectionResult:
    result = results[0]
    annotated_frame = result.plot()

    if len(result.boxes) == 0:
        return DetectionResult(
            class_id=None,
            conf=None,
            detection=False,
            annotated_frame=annotated_frame
        )

    else:
        best_idx = result.boxes.conf.argmax()
    
        class_id = int(result.boxes.cls[best_idx])
    
        conf = float(result.boxes.conf[best_idx])

    return DetectionResult(
        class_id=class_id,
        conf=conf,
        detection=True,
        annotated_frame=annotated_frame
    )


def render_detection_result(result: DetectionResult):
    col1, col2 = st.columns(2)

    with col1:
        st.image(result.annotated_frame, channels="BGR")
    
    with col2:
        if result.detection:
            info = disease_info.get(result.class_id)
    
            st.subheader(info["explain"])
    
            st.progress(result.conf)
    
            st.write(f"신뢰도: {result.conf:.2f}")
    
        else:
            st.subheader("탐지된 병해충이 없습니다.")
            st.success("건강한 딸기로 보입니다 🍓")


def get_video_info(video_path : str) -> VideoInfo:

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    duration = total_frames / fps

    cap.release()

    return VideoInfo(fps=fps,
                     width=width,
                     height=height,
                     total_frames=total_frames,
                     duration=duration)
