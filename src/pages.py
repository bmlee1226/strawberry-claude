import streamlit as st
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

import cv2
import av
import numpy as np
import threading
import os
import tempfile
from datetime import datetime

from src import process
from src import utility
from src.disease_data import disease_info


class _MockVideoFile:
    """실시간 촬영 후 분석 흐름에서 uploaded_file 역할을 대신하는 객체."""
    type = "video/mp4"


# STUN + 복수 TURN 서버 (TCP/UDP, 80/443 포트 모두 포함 — 방화벽 환경 대응)
_RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": [
                "turn:openrelay.metered.ca:80",
                "turn:openrelay.metered.ca:443",
                "turn:openrelay.metered.ca:443?transport=tcp",
            ],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
    ]
})

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()


def page_home():
  st.title("🍓 딸기 병해충 진단 AI")
  
  st.markdown("""
  딸기 이미지를 업로드하면  
  AI가 병해충을 탐지하고 원인 및 해결 방법을 안내합니다.
  """)
  
  st.info("현재 지원: 흰가루병, 잿빛곰팡이병")
  
  st.warning("""
  본 결과는 AI 예측이며
  정확한 진단은 전문가 확인이 필요합니다.
  """)
  
  st.subheader("예시 이미지")
  
  col_a, col_b, col_c = st.columns(3)
  
  with col_a:
      st.image("gray_mold.png", use_container_width=True)
  
  with col_b:
      st.image("powdery_mildew.jpg", use_container_width=True)
  
  with col_c:
      st.image("healthy.png", use_container_width=True)
  
  with st.sidebar:
  
      st.header("⚙️ 설정")
  
      st.divider()
  
      conf_threshold = st.slider(
          "신뢰도 임계값 (Confidence Threshold)",
          min_value=0.1,
          max_value=1.0,
          value=0.3,
          step=0.05
      )
  
      st.caption("""
          값이 낮을수록 더 많은 병해를 탐지하지만
  오탐 가능성이 증가할 수 있습니다.
          """)
  
  st.session_state.conf_threshold = conf_threshold
  
  st.divider()

  st.markdown(
      "<h3 style='text-align:center; margin-bottom: 0.2rem;'>📌 분석 방식을 선택하세요</h3>",
      unsafe_allow_html=True
  )
  st.markdown(
      "<p style='text-align:center; color:gray; margin-bottom: 1.5rem;'>아래 카드 중 하나를 클릭해 시작하세요</p>",
      unsafe_allow_html=True
  )

  colum1, colum2, colum3 = st.columns(3)

  # -----------------------------------
  # 이미지 분석 카드
  # -----------------------------------

  with colum1:
      with st.container(border=True):
          st.markdown("## 🖼️")
          st.subheader("이미지 분석")
          st.write("딸기 사진을 업로드하거나 카메라로 촬영하면 AI가 즉시 병해충을 진단합니다.")
          st.caption("✅ 빠른 분석 · 사진 1장")
          st.markdown("<br>", unsafe_allow_html=True)
          if st.button(
              "👉 이미지 분석 시작하기",
              use_container_width=True,
              type="primary"
          ):
              go_to("image")

  # -----------------------------------
  # 동영상 분석 카드
  # -----------------------------------

  with colum2:
      with st.container(border=True):
          st.markdown("## 🎥")
          st.subheader("동영상 분석")
          st.write("딸기 재배 영상을 업로드하면 AI가 전체 구간에서 병해충을 탐지합니다.")
          st.caption("✅ 넓은 구역 탐지 · MP4/AVI/MOV")
          st.markdown("<br>", unsafe_allow_html=True)
          if st.button(
              "👉 동영상 분석 시작하기",
              use_container_width=True,
              type="primary"
          ):
              go_to("video")

  # -----------------------------------
  # 실시간 동영상 카드
  # -----------------------------------

  with colum3:
      with st.container(border=True):
          st.markdown("## 📹")
          st.subheader("실시간 촬영 분석")
          st.write("카메라로 직접 딸기를 촬영하고 바로 병해충을 분석합니다.")
          st.caption("✅ 현장 즉시 촬영 · 빠른/정밀 분석")
          st.markdown("<br>", unsafe_allow_html=True)
          if st.button(
              "👉 실시간 촬영 시작하기",
              use_container_width=True,
              type="primary"
          ):
              go_to("realtime_video")

def page_image():
  
  st.title("🖼 이미지 병해충 분석")
  
  colum1, colum2 = st.columns(2)
  
  with colum1:
      
      uploaded_file = st.file_uploader("이미지 업로드")
  
  with colum2:
  
      camera_image = st.camera_input("사진 촬영")
  
  if uploaded_file:
      st.session_state.uploaded_file = uploaded_file
  
      st.success("✅ 이미지 업로드 완료")
  
      # 결과 페이지로 이동
      go_to("analysis")
  
  elif camera_image:
      st.session_state.uploaded_file = camera_image
  
      st.success("✅ 이미지 업로드 완료")
  
      # 결과 페이지로 이동
      go_to("analysis")

def _render_video_analysis_options(video_path):
    """영상 정보 표시 및 분석 방식 선택 UI (업로드/실시간 공용)."""
    videoinfo = utility.get_video_info(video_path)

    st.subheader("영상 정보")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("FPS", f"{videoinfo.fps:.1f}")
    with col2:
        st.metric("총 프레임", videoinfo.total_frames)
    with col3:
        st.metric("영상 길이", f"{videoinfo.duration:.1f}초")

    FAST_FPS = 15
    PRECISE_FPS = 5
    fast_estimated_time = int(videoinfo.duration) / FAST_FPS
    precise_estimated_time = videoinfo.total_frames / PRECISE_FPS

    st.subheader("분석 방식 선택")
    col1, col2 = st.columns(2)

    with col1:
        st.info(
            f"""
            빠른 분석

            • 1초당 1프레임 분석
            • 긴 영상 빠른 확인용
            • 예상 시간: {fast_estimated_time:.1f}초
            """
        )
        if st.button("빠른 분석 시작", use_container_width=True):
            st.session_state.analysis_type = "fast"
            go_to("analysis")

    with col2:
        st.warning(
            f"""
            정밀 분석

            • 모든 프레임 분석
            • 가장 정확한 결과
            • 결과 mp4 생성
            • 예상 시간: {precise_estimated_time:.1f}초
            """
        )
        if st.button("정밀 분석 시작", use_container_width=True):
            st.session_state.analysis_type = "precise"
            go_to("analysis")


def page_video():

    st.title("🎥 동영상 병해충 분석")

    # 실시간 촬영으로 넘어온 경우 — 업로더 없이 분석 옵션 바로 표시
    video_path = st.session_state.get("video_path")
    if video_path and isinstance(st.session_state.get("uploaded_file"), _MockVideoFile):
        st.success("✅ 촬영된 동영상이 준비되었습니다.")
        with open(video_path, "rb") as f:
            st.video(f.read())
        _render_video_analysis_options(video_path)
        return

    uploaded_video_file = st.file_uploader(
        "동영상을 업로드하세요",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video_file is not None:

        video_bytes = uploaded_video_file.read()
        st.session_state.uploaded_file = uploaded_video_file

        st.success("✅ 동영상 업로드 완료")

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_bytes)
        tfile.close()

        st.session_state.video_path = tfile.name

        st.video(video_bytes)
        _render_video_analysis_options(tfile.name)

def page_realtime_video():

    st.title("📹 실시간 동영상 촬영")
    st.write("카메라로 딸기를 촬영한 뒤 **녹화 중지 및 분석**을 눌러 분석 방식을 선택하세요.")

    class _VideoRecorder(VideoProcessorBase):
        def __init__(self):
            self.recording = False
            self.frames: list = []
            self._lock = threading.Lock()

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            with self._lock:
                if self.recording:
                    self.frames.append(img.copy())
            return frame

    ctx = webrtc_streamer(
        key="realtime_recorder",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=_RTC_CONFIG,
        video_processor_factory=_VideoRecorder,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔴 녹화 시작", use_container_width=True, type="primary"):
                with ctx.video_processor._lock:
                    ctx.video_processor.frames = []
                ctx.video_processor.recording = True
                st.info("🔴 녹화 중입니다...")

        with col2:
            if st.button("⏹ 녹화 중지 및 분석", use_container_width=True):
                ctx.video_processor.recording = False

                with ctx.video_processor._lock:
                    frames = list(ctx.video_processor.frames)

                if not frames:
                    st.warning("녹화된 프레임이 없습니다. 먼저 녹화를 시작하세요.")
                else:
                    h, w = frames[0].shape[:2]
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tfile.close()

                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter(tfile.name, fourcc, 20.0, (w, h))
                    for f in frames:
                        out.write(f)
                    out.release()

                    st.session_state.video_path = tfile.name
                    st.session_state.uploaded_file = _MockVideoFile()
                    st.session_state.analysis_result = None
                    st.session_state.analysis_type = None

                    go_to("video")


def page_analysis():

  if st.session_state.analysis_result is None:
      st.title("📊 분석 중")
      
      uploaded_file = st.session_state.uploaded_file
      conf_threshold = st.session_state.conf_threshold
      
      file_type = uploaded_file.type
      
      # 이미지인 경우
      if "image" in file_type:
      
          analysis_result = process.process_image(uploaded_file, model, conf_threshold)
      
      # 동영상인 경우
      elif "video" in file_type:
      
          if st.session_state.analysis_type == "fast":
              
              video_path = st.session_state.video_path
              analysis_result = process.process_fast_video(video_path, model, conf_threshold)
      
                      
          elif st.session_state.analysis_type == "precise":
      
              video_path = st.session_state.video_path
              analysis_result = process.process_precise_video(video_path, model, conf_threshold)
    
      st.session_state.analysis_result = analysis_result
      
  go_to("result")
  

def _render_video_detection_summary(analysis_result):
    """동영상 분석 결과 요약 카드를 렌더링한다."""

    st.divider()
    st.header("📊 병해충 탐지 결과")
    st.caption(f"신뢰도 임계값: {analysis_result.conf_threshold}")

    total_frames = len(analysis_result.result_list) or analysis_result.detection_frame_count
    detected = analysis_result.detection_frame_count
    healthy = total_frames - detected if total_frames else 0

    # ------- 상단 요약 지표 -------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎞 분석 프레임", f"{total_frames}프레임")
    with col2:
        st.metric("🦠 병해 탐지 프레임", f"{detected}프레임",
                  delta=f"{detected/total_frames*100:.1f}%" if total_frames else None,
                  delta_color="inverse")
    with col3:
        st.metric("✅ 정상 프레임", f"{healthy}프레임")

    st.divider()

    if detected == 0:
        st.success("✅ 병해충이 탐지되지 않았습니다. 현재 상태를 유지하세요.")
        return

    # ------- 클래스별 탐지 비율 카드 -------
    st.subheader("🔍 클래스별 탐지 현황")

    counts = analysis_result.detected_class_counts
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    cols = st.columns(len(sorted_counts))
    for col, (class_id, count) in zip(cols, sorted_counts):
        name = disease_info[class_id]["name"]
        ratio = count / total_frames * 100 if total_frames else 0
        with col:
            with st.container(border=True):
                st.markdown(f"### {count}프레임")
                st.write(f"**{name}**")
                st.progress(ratio / 100)
                st.caption(f"전체의 {ratio:.1f}%")

    st.divider()

    # ------- 병해 상세 정보 -------
    st.subheader("📋 병해 상세 정보")
    for class_id, _ in sorted_counts:
        utility.show_disease_info(class_id)


_DEVELOPER_PASSWORD = "strawberry-dev-2024"

_LABEL_DIR_MAP = {
    "정상": "healthy",
    "흰가루병": "powdery_mildew",
    "잿빛곰팡이병": "gray_mold",
}


def _save_training_image(uploaded_file, label: str) -> None:
    save_dir = os.path.join("user_uploads", _LABEL_DIR_MAP[label])
    os.makedirs(save_dir, exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
    save_path = os.path.join(save_dir, filename)
    uploaded_file.seek(0)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"✅ [{label}] 라벨로 저장되었습니다.")


def _render_developer_data_viewer() -> None:
    st.divider()
    st.subheader("🔐 개발자 — 학습 데이터 현황")

    base_dir = "user_uploads"
    if not os.path.exists(base_dir):
        st.info("아직 저장된 학습 데이터가 없습니다.")
        return

    total = 0
    label_counts = {}
    for label, folder in _LABEL_DIR_MAP.items():
        folder_path = os.path.join(base_dir, folder)
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            label_counts[label] = files
            total += len(files)
        else:
            label_counts[label] = []

    # 요약 지표
    cols = st.columns(len(_LABEL_DIR_MAP) + 1)
    cols[0].metric("전체 이미지", total)
    for i, (label, files) in enumerate(label_counts.items(), 1):
        cols[i].metric(label, len(files))

    st.divider()

    # 라벨별 이미지 탐색
    selected = st.selectbox(
        "라벨 선택",
        list(_LABEL_DIR_MAP.keys()),
        key="dev_label_select"
    )
    files = label_counts[selected]

    if not files:
        st.info(f"[{selected}] 라벨에 저장된 이미지가 없습니다.")
        return

    st.caption(f"{len(files)}개 이미지")

    # 한 행에 4개씩 표시
    cols_per_row = 4
    for i in range(0, len(files), cols_per_row):
        row_files = files[i:i + cols_per_row]
        row_cols = st.columns(cols_per_row)
        for col, fname in zip(row_cols, row_files):
            path = os.path.join(base_dir, _LABEL_DIR_MAP[selected], fname)
            col.image(path, caption=fname, use_container_width=True)


def _render_developer_sidebar() -> None:
    with st.sidebar:
        st.divider()
        st.subheader("🔐 개발자 모드")
        if st.session_state.get("is_developer"):
            st.success("개발자 모드 활성화됨")
            if st.button("로그아웃", use_container_width=True):
                st.session_state.is_developer = False
                st.rerun()
        else:
            pw = st.text_input("비밀번호", type="password", key="dev_pw_input")
            if st.button("로그인", use_container_width=True):
                if pw == _DEVELOPER_PASSWORD:
                    st.session_state.is_developer = True
                    st.rerun()
                else:
                    st.error("비밀번호가 틀렸습니다.")


def page_result():

    uploaded_file = st.session_state.uploaded_file
    file_type = uploaded_file.type
    analysis_result = st.session_state.analysis_result
    
    if "image" in file_type:
        result_list = analysis_result.result_list
        detection_result = result_list[0]

        utility.render_detection_result(detection_result)

        if detection_result.detection:
            utility.show_disease_info(detection_result.class_id)

        # ------- 사용자: 피드백 및 학습 데이터 저장 -------
        st.divider()
        st.subheader("📦 AI 진단 피드백")
        st.write("AI 진단 결과가 맞는지 확인하고, 올바른 라벨로 저장해 모델 개선에 기여하세요.")

        LABEL_OPTIONS = ["정상", "흰가루병", "잿빛곰팡이병"]

        if detection_result.detection:
            class_to_label = {0: "잿빛곰팡이병", 1: "흰가루병"}
            default_label = class_to_label.get(detection_result.class_id, "정상")
        else:
            default_label = "정상"

        selected_label = st.selectbox(
            "실제 병해 라벨을 선택하세요",
            LABEL_OPTIONS,
            index=LABEL_OPTIONS.index(default_label)
        )

        if st.button("💾 학습 데이터로 저장", type="primary", use_container_width=True):
            _save_training_image(uploaded_file, selected_label)

        # ------- 개발자 전용: 저장된 학습 데이터 뷰어 -------
        if st.session_state.get("is_developer"):
            _render_developer_data_viewer()

    
    elif "video" in file_type:
        if st.session_state.analysis_type == "fast":
            result_list = analysis_result.result_list
            for detection_result in result_list:
                utility.render_detection_result(detection_result)

        elif st.session_state.analysis_type == "precise":
            st.video(analysis_result.final_output)
            with open(analysis_result.final_output, "rb") as file:
                st.download_button(
                    label="결과 영상 다운로드",
                    data=file,
                    file_name="result.mp4",
                    mime="video/mp4"
                )

        _render_video_detection_summary(analysis_result)
        
    if st.button("🔙 처음으로"):

        temp_output = analysis_result.temp_output
    
        if temp_output and os.path.exists(temp_output):
    
            os.remove(temp_output)

        video_path = st.session_state.get(
            "video_path"
        )
        
        if video_path and os.path.exists(video_path):
        
            os.remove(video_path)
    
        st.session_state.uploaded_file = None
        st.session_state.video_path = None
        st.session_state.analysis_result = None
        st.session_state.analysis_type = None
          
        go_to("home")
  
      


def go_to(page):

    st.session_state.page = page

    st.rerun()
