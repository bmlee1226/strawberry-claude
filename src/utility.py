import cv2
import streamlit as st

from src.disease_data import disease_info
from src.data_models import DetectionResult, VideoInfo

_RISK_BADGE = {
    "high":   ("🔴", "위험"),
    "medium": ("🟡", "주의"),
    "none":   ("🟢", "정상"),
}


def show_disease_info(class_id) -> None:
    info = disease_info.get(class_id)
    if not info:
        return

    icon, label = _RISK_BADGE.get(info.get("risk", "none"), ("⚪", ""))

    st.markdown(f"## {icon} {info['name']} <sup style='font-size:0.6em; color:gray;'>{label}</sup>",
                unsafe_allow_html=True)

    # 예시 이미지 + 탭 나란히
    col_img, col_detail = st.columns([1, 2])

    with col_img:
        st.image(info["image"], use_container_width=True)

    with col_detail:
        pesticides = info.get("pesticides", [])
        tabs = st.tabs(["🍃 증상", "🦠 원인", "💊 해결책", "🧴 방제 약품"])

        with tabs[0]:
            st.markdown(info["symptom"])

        with tabs[1]:
            st.markdown(info["cause"])

        with tabs[2]:
            for item in info["solution"]:
                st.markdown(f"- {item}")

        with tabs[3]:
            if not pesticides:
                st.info("건강한 상태입니다. 약품이 필요하지 않습니다.")
            else:
                st.caption("⚠️ 반드시 라벨 지시에 따라 사용하고, 안전사용 기준을 지켜주세요.")
                for p in pesticides:
                    with st.container(border=True):
                        col_a, col_b, col_c = st.columns([3, 2, 2])
                        col_a.markdown(f"**{p['name']}**")
                        col_b.caption(f"🗓 {p['timing']}")
                        col_c.caption(f"🔢 {p['limit']}")


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
