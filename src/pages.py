import streamlit as st
import streamlit.components.v1 as components
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

import cv2
import av
import numpy as np
import threading
import io
import zipfile
import os
import tempfile
from datetime import datetime

from src import process
from src import utility
from src import history as hist
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


def page_name_input():
    st.markdown("""
    <div style='text-align:center; padding:2rem 0 1rem;'>
      <div style='font-size:3rem;'>🍓</div>
      <h1 style='font-size:1.9rem; margin:0.4rem 0;'>딸기 병 진단기</h1>
      <p style='font-size:1.15rem; color:#555;'>시작하기 전에 이름을 알려주세요.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("### 👤 이름 또는 농장 이름을 입력하세요")
        st.markdown("입력한 이름으로 진단 이력이 저장됩니다.")
        name = st.text_input(
            "이름",
            placeholder="예: 홍길동 / 행복농장",
            label_visibility="collapsed",
            key="name_input_field",
        )
        if st.button("✅ 시작하기", use_container_width=True, type="primary"):
            if name.strip():
                st.session_state.user_name = name.strip()
                go_to("home")
            else:
                st.warning("이름을 입력해 주세요.")


def page_home():

    # 예시 이미지 동일 크기 고정 CSS
    st.markdown("""
    <style>
    .example-img-wrap img {
        width: 100%;
        height: 220px;
        object-fit: cover;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # 히어로 섹션
    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 0.5rem 0;'>
      <div style='font-size:3rem;'>🍓</div>
      <h1 style='font-size:2rem; margin:0.3rem 0;'>딸기 병 진단기</h1>
      <p style='font-size:1.2rem; color:#555; margin:0;'>
        딸기 사진이나 영상을 올리면<br>
        <b>AI가 병을 확인하고 해결 방법을 알려드립니다.</b>
      </p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_m, col_r = st.columns(3)
    col_m.info("🌿 흰가루병 · 잿빛곰팡이병 진단 가능")
    st.caption("⚠️ AI 예측 결과이며, 정확한 진단은 전문가 확인이 필요합니다.")

    entries = hist.load_all(user_name=st.session_state.get("user_name", ""))
    if entries:
        col_h1, col_h2 = st.columns([3, 1])
        last = entries[0]
        col_h1.markdown(f"📋 마지막 진단: **{last.get('date','')}** — {'🚨 ' + (last.get('disease_name') or ', '.join(last.get('disease_names', []))) if last.get('detected') else '✅ 정상'}")
        if col_h2.button("이력 보기", use_container_width=True):
            go_to("history")

    # 사이드바
    with st.sidebar:
        st.markdown("### ⚙️ 민감도 설정")
        st.markdown("숫자가 낮을수록 병을 더 많이 찾습니다.<br>처음엔 **0.30** 그대로 두세요.", unsafe_allow_html=True)
        conf_threshold = st.slider(
            "민감도",
            min_value=0.1,
            max_value=1.0,
            value=st.session_state.get("conf_threshold", 0.3),
            step=0.05,
            label_visibility="collapsed",
        )
        st.markdown(f"<p style='text-align:center; font-size:1.3rem; font-weight:bold;'>현재: {conf_threshold:.2f}</p>", unsafe_allow_html=True)

    st.session_state.conf_threshold = conf_threshold
    st.divider()

    # 시작 방법 선택
    st.markdown("""
    <h2 style='text-align:center; margin-bottom:0.3rem;'>어떻게 진단할까요?</h2>
    <p style='text-align:center; color:#666; font-size:1.05rem; margin-bottom:1.2rem;'>
      아래 세 가지 중 하나를 골라 누르세요.
    </p>
    """, unsafe_allow_html=True)

    colum1, colum2, colum3 = st.columns(3)

    with colum1:
        with st.container(border=True):
            st.markdown("<div style='font-size:2.5rem; text-align:center;'>📷</div>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align:center; margin:0.3rem 0;'>사진으로 진단</h3>", unsafe_allow_html=True)
            st.markdown("<p style='text-align:center; color:#555; font-size:0.95rem;'>찍은 사진을 올리거나<br>지금 바로 찍어주세요</p>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📷 사진으로 시작", use_container_width=True, type="primary"):
                go_to("image")

    with colum2:
        with st.container(border=True):
            st.markdown("<div style='font-size:2.5rem; text-align:center;'>🎥</div>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align:center; margin:0.3rem 0;'>동영상으로 진단</h3>", unsafe_allow_html=True)
            st.markdown("<p style='text-align:center; color:#555; font-size:0.95rem;'>찍어둔 동영상을<br>올려주세요</p>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🎥 동영상으로 시작", use_container_width=True, type="primary"):
                go_to("video")

    with colum3:
        with st.container(border=True):
            st.markdown("<div style='font-size:2.5rem; text-align:center;'>📹</div>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align:center; margin:0.3rem 0;'>직접 찍어서 진단</h3>", unsafe_allow_html=True)
            st.markdown("<p style='text-align:center; color:#555; font-size:0.95rem;'>카메라로 딸기밭을 천천히<br>이동하며 촬영하면<br>넓은 범위를 한 번에 확인!</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align:center; color:#e07000; font-size:0.88rem;'>🌿 밭 전체를 둘러보고 싶을 때 추천</p>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📹 직접 찍어서 시작", use_container_width=True, type="primary"):
                go_to("realtime_video")

    st.divider()

    # 예시 이미지
    st.markdown("<h3 style='text-align:center;'>📷 이런 병이 있을 때 사용하세요</h3>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown('<div class="example-img-wrap">', unsafe_allow_html=True)
        st.image("gray_mold.png", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; font-size:1rem; font-weight:bold; color:#cc3300;'>🔴 잿빛곰팡이병</p>", unsafe_allow_html=True)
        st.caption("과실에 회색 곰팡이가 핌")

    with col_b:
        st.markdown('<div class="example-img-wrap">', unsafe_allow_html=True)
        st.image("powdery_mildew.jpg", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; font-size:1rem; font-weight:bold; color:#cc8800;'>🟡 흰가루병</p>", unsafe_allow_html=True)
        st.caption("잎에 흰 가루가 덮임")

    with col_c:
        st.markdown('<div class="example-img-wrap">', unsafe_allow_html=True)
        st.image("healthy.png", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; font-size:1rem; font-weight:bold; color:#117733;'>🟢 건강한 상태</p>", unsafe_allow_html=True)
        st.caption("이상 없음")

def page_image():

    st.title("📷 사진으로 진단하기")
    st.markdown("<p style='font-size:1.15rem; color:#444;'>딸기 사진을 찍거나 올리면 AI가 병을 확인해 드립니다.</p>", unsafe_allow_html=True)

    st.markdown("""
<div class='senior-step'>
  <span class='senior-step-num'>1</span> 아래 두 가지 방법 중 <b>하나</b>를 선택하세요.
</div>
<div class='senior-step'>
  <span class='senior-step-num'>2</span> 사진을 올리거나 찍으면 <b>자동으로 분석이 시작</b>됩니다.
</div>
<div class='senior-step'>
  <span class='senior-step-num'>3</span> 잠시 기다리면 결과를 알려드립니다.
</div>
<div class='senior-tip'>💡 딸기 잎이나 열매가 화면에 크게 나오도록 가까이서 찍으면 더 정확합니다.</div>
""", unsafe_allow_html=True)

    st.divider()

    notice = st.empty()

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("### 📁 사진 파일 올리기")
            st.markdown("스마트폰 사진첩에서 사진을 선택하세요.")
            uploaded_file = st.file_uploader(
                "사진 올리기",
                type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
                label_visibility="collapsed",
                key=f"img_uploader_{st.session_state.get('img_upload_key', 0)}",
            )
            st.caption("👆 위 버튼을 누르거나 사진을 끌어다 놓으세요.")

    with col2:
        with st.container(border=True):
            st.markdown("### 📸 지금 바로 찍기")
            st.markdown("카메라로 딸기를 찍어주세요.")
            components.html("""
<script>
(function() {
  const orig = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia = function(constraints) {
    if (constraints && constraints.video) {
      constraints.video = typeof constraints.video === 'object'
        ? { ...constraints.video, facingMode: 'environment' }
        : { facingMode: 'environment' };
    }
    return orig(constraints);
  };
})();
</script>
""", height=0)
            camera_image = st.camera_input(
                "사진 찍기",
                label_visibility="collapsed",
            )
            st.caption("📷 화면 아래 동그란 버튼을 누르면 사진이 찍힙니다.")

    if uploaded_file:
        if "video" in uploaded_file.type:
            with notice.container():
                st.warning("🎥 동영상 파일입니다. 동영상 분석 페이지를 이용해주세요.")
                col_v, col_r = st.columns(2)
                with col_v:
                    if st.button("🎥 동영상 분석으로 이동", use_container_width=True, type="primary"):
                        go_to("video")
                with col_r:
                    if st.button("🔙 다시 올리기", use_container_width=True):
                        st.session_state.img_upload_key = st.session_state.get("img_upload_key", 0) + 1
                        st.rerun()
        else:
            st.session_state.uploaded_file = uploaded_file
            st.success("✅ 사진이 올라갔습니다! 잠시만 기다려 주세요...")
            go_to_top("analysis")

    elif camera_image:
        st.session_state.uploaded_file = camera_image
        st.success("✅ 사진을 찍었습니다! 잠시만 기다려 주세요...")
        go_to_top("analysis")

def _render_video_analysis_options(video_path):
    """영상 정보 표시 및 분석 방식 선택 UI (업로드/실시간 공용)."""
    videoinfo = utility.get_video_info(video_path)

    st.caption(
        f"📋 영상 정보 — 길이: {videoinfo.duration:.1f}초 · 프레임: {videoinfo.total_frames}장 · FPS: {videoinfo.fps:.1f}"
    )

    FAST_FPS = 15
    PRECISE_FPS = 20
    fast_estimated_time = int(videoinfo.duration) / FAST_FPS
    precise_estimated_time = videoinfo.total_frames / PRECISE_FPS

    st.markdown("""
    <div class='step-bar' style='display:flex; gap:0.5rem; align-items:center; margin: 1rem 0 0.5rem 0;'>
      <span style='background:#eee; color:#aaa; padding:4px 14px; border-radius:20px;'>1 동영상 업로드</span>
      <span style='color:#ccc;'>→</span>
      <span style='background:#FF4B4B; color:white; padding:4px 14px; border-radius:20px; font-weight:bold;'>2 분석 방식 선택</span>
      <span style='color:#ccc;'>→</span>
      <span style='background:#eee; color:#aaa; padding:4px 14px; border-radius:20px;'>3 AI 분석</span>
      <span style='color:#ccc;'>→</span>
      <span style='background:#eee; color:#aaa; padding:4px 14px; border-radius:20px;'>4 결과 확인</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    st.markdown("<p style='font-size:1.1rem; font-weight:bold; margin-bottom:0.5rem;'>아래 두 가지 중 하나를 눌러주세요.</p>", unsafe_allow_html=True)

    with col1:
        with st.container(border=True):
            st.markdown("### ⚡ 빨리 확인하기")
            st.markdown("결과를 빠르게 보고 싶을 때")
            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("소요 시간", f"약 {fast_estimated_time:.0f}초")
            c2.metric("분석 장수", f"{int(videoinfo.duration)}장")
            st.markdown("- 영상을 빠르게 훑어봅니다\n- 병해 여부를 빠르게 확인")
            if st.button("⚡ 빨리 확인하기", use_container_width=True, type="primary"):
                st.session_state.analysis_type = "fast"
                go_to("analysis")

    with col2:
        with st.container(border=True):
            st.markdown("### 🔬 꼼꼼히 확인하기")
            st.markdown("정확한 결과가 필요할 때")
            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("소요 시간", f"약 {precise_estimated_time:.0f}초")
            c2.metric("분석 장수", f"{videoinfo.total_frames}장")
            st.markdown("- 영상 전체를 꼼꼼히 분석합니다\n- 결과 영상도 저장됩니다")
            if st.button("🔬 꼼꼼히 확인하기", use_container_width=True, type="primary"):
                st.session_state.analysis_type = "precise"
                go_to("analysis")


def page_video():

    st.title("🎥 동영상으로 진단하기")
    st.markdown("<p style='font-size:1.15rem; color:#444;'>딸기 동영상을 올리면 AI가 병을 확인해 드립니다.</p>", unsafe_allow_html=True)

    # 실시간 촬영으로 넘어온 경우
    video_path = st.session_state.get("video_path")
    if video_path and isinstance(st.session_state.get("uploaded_file"), _MockVideoFile):
        st.success("✅ 촬영된 동영상이 준비되었습니다.")
        _render_video_analysis_options(video_path)
        return

    st.markdown("""
<div class='senior-step'>
  <span class='senior-step-num'>1</span> 아래 버튼을 눌러 <b>동영상 파일을 선택</b>하세요.
</div>
<div class='senior-step'>
  <span class='senior-step-num'>2</span> 파일을 올리면 <b>분석 방법 선택 화면</b>이 나타납니다.
</div>
<div class='senior-step'>
  <span class='senior-step-num'>3</span> <b>빠른 분석</b> 또는 <b>꼼꼼히 분석</b> 중 하나를 누르세요.
</div>
<div class='senior-tip'>💡 딸기 재배 구역을 천천히 움직이며 찍은 영상이 가장 좋습니다. (30초~3분 권장)</div>
""", unsafe_allow_html=True)

    st.divider()

    notice = st.empty()

    with st.container(border=True):
        st.markdown("### 📁 동영상 파일 올리기")
        st.markdown("스마트폰에서 찍은 동영상을 선택하세요.")
        uploaded_video_file = st.file_uploader(
            "동영상 올리기",
            type=["mp4", "avi", "mov", "jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key=f"vid_uploader_{st.session_state.get('vid_upload_key', 0)}",
        )
        st.caption("👆 위 버튼을 누르면 파일을 선택할 수 있습니다.")

    if uploaded_video_file is not None:

        if "image" in uploaded_video_file.type:
            with notice.container():
                st.warning("🖼 사진 파일입니다. 사진 분석 페이지를 이용해주세요.")
                col_i, col_r = st.columns(2)
                with col_i:
                    if st.button("📷 사진 분석으로 이동", use_container_width=True, type="primary"):
                        go_to("image")
                with col_r:
                    if st.button("🔙 다시 올리기", use_container_width=True):
                        st.session_state.vid_upload_key = st.session_state.get("vid_upload_key", 0) + 1
                        st.rerun()
        else:
            video_bytes = uploaded_video_file.read()
            st.session_state.uploaded_file = uploaded_video_file
            st.success("✅ 동영상이 올라갔습니다! 아래에서 분석 방법을 선택해 주세요.")
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_bytes)
            tfile.close()
            st.session_state.video_path = tfile.name
            _render_video_analysis_options(tfile.name)

def page_realtime_video():

    st.title("📹 직접 찍어서 진단하기")
    st.markdown("<p style='font-size:1.15rem; color:#444;'>카메라로 딸기밭을 천천히 이동하며 찍으면 <b>넓은 범위를 한 번에</b> 확인할 수 있습니다.</p>", unsafe_allow_html=True)

    st.markdown("""
<div style='background:#fff8e1; border:2px solid #f5c518; border-radius:12px;
     padding:1rem 1.2rem; margin-bottom:1rem;'>
  <p style='font-size:1.05rem; margin:0; color:#5a4000;'>
    🌿 <b>이럴 때 사용하세요</b><br>
    딸기 한 포기가 아닌 밭 전체를 확인하고 싶을 때,<br>
    카메라를 들고 천천히 걸으면서 찍으면 AI가 병든 곳을 찾아드립니다.
  </p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class='senior-step'>
  <span class='senior-step-num'>1</span> 아래 <b>START</b> 버튼을 누르세요. <span style='color:#888; font-size:0.95rem;'>(카메라가 켜지는 데 5초 정도 걸립니다)</span>
</div>
<div class='senior-step'>
  <span class='senior-step-num'>2</span> 카메라 화면이 나오면 딸기밭을 향해 들어주세요.
</div>
<div class='senior-step'>
  <span class='senior-step-num'>3</span> 빨간 <b>🔴 녹화 시작</b> 버튼을 누르고, 딸기밭을 <b>천천히 걸으며</b> 찍어주세요.
</div>
<div class='senior-step'>
  <span class='senior-step-num'>4</span> 다 찍었으면 <b>⏹ 촬영 완료</b> 버튼을 누르세요.
</div>
<div class='senior-tip'>💡 카메라가 켜지지 않으면 아래 안내를 확인하세요.</div>
""", unsafe_allow_html=True)

    with st.expander("📷 카메라가 켜지지 않거나 앞 카메라가 나올 때  ▼ 탭하여 펼치기"):
        st.markdown("""
1. 화면 주소창 옆 **자물쇠(🔒) 아이콘**을 눌러주세요.
2. **카메라** 항목을 찾아 **허용**으로 바꾸세요.
3. 페이지를 **새로고침**해서 다시 시도하세요.
        """)

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
        media_stream_constraints={
            "video": {"facingMode": "environment"},
            "audio": False,
        },
        async_processing=True,
    )

    if ctx.video_processor:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔴 녹화 시작", use_container_width=True, type="primary"):
                with ctx.video_processor._lock:
                    ctx.video_processor.frames = []
                ctx.video_processor.recording = True
                st.info("🔴 녹화 중입니다. 딸기를 향해 카메라를 들어주세요.")

        with col2:
            if st.button("⏹ 촬영 완료", use_container_width=True):
                ctx.video_processor.recording = False

                with ctx.video_processor._lock:
                    frames = list(ctx.video_processor.frames)

                if not frames:
                    st.warning("촬영된 영상이 없습니다. 먼저 🔴 녹화 시작을 눌러주세요.")
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

        if "image" in file_type:
            analysis_result = process.process_image(uploaded_file, model, conf_threshold)

        elif "video" in file_type:
            video_path = st.session_state.video_path
            if st.session_state.analysis_type == "fast":
                analysis_result = process.process_fast_video(video_path, model, conf_threshold)
            elif st.session_state.analysis_type == "precise":
                analysis_result = process.process_precise_video(video_path, model, conf_threshold)

        st.session_state.analysis_result = analysis_result
        st.session_state._result_saved = False
        st.session_state._result_save_status = None

    go_to_top("result")
  

def _render_video_detection_summary(analysis_result, compact: bool = False):
    """동영상 분석 결과 요약 카드를 렌더링한다.
    compact=True 이면 헤더·divider 없이 컬럼 안에 들어갈 수 있는 형태로 렌더링.
    """
    total_frames = len(analysis_result.result_list) or analysis_result.detection_frame_count
    detected = analysis_result.detection_frame_count
    healthy = total_frames - detected if total_frames else 0

    if not compact:
        st.divider()
        st.header("📊 분석 결과")

    st.caption(f"민감도 설정값: {analysis_result.conf_threshold}")

    # ------- 요약 지표 -------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎞 살펴본 장면 수", f"{total_frames}장")
    with col2:
        st.metric("🦠 병 발견 장면", f"{detected}장",
                  delta=f"{detected/total_frames*100:.1f}%" if total_frames else None,
                  delta_color="inverse")
    with col3:
        st.metric("✅ 이상 없는 장면", f"{healthy}장")

    if detected == 0:
        st.success("병이 발견되지 않았습니다.")
        return []

    st.divider()

    # ------- 클래스별 탐지 비율 카드 -------
    st.markdown("**어떤 병이 얼마나 발견됐나요?**")
    counts = analysis_result.detected_class_counts
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    cols = st.columns(len(sorted_counts))
    for col, (class_id, count) in zip(cols, sorted_counts):
        name = disease_info[class_id]["name"]
        ratio = count / total_frames * 100 if total_frames else 0
        with col:
            with st.container(border=True):
                st.markdown(f"### {count}장 발견")
                st.write(f"**{name}**")
                st.progress(ratio / 100)
                st.caption(f"전체 장면의 {ratio:.1f}% 에서 발견됨")

    # ------- 병해 상세 정보 (compact 모드에서는 생략 — 호출부에서 별도 렌더링) -------
    if not compact:
        st.divider()
        with st.expander("📖 이 병은 무엇인가요? (원인과 대처 방법 보기)  ▼ 탭하여 펼치기", expanded=False):
            for class_id, _ in sorted_counts:
                utility.show_disease_info(class_id)

    return sorted_counts


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
    st.success("✅ 저장이 완료되었습니다. AI 성능 향상에 기여해 주셔서 감사합니다! 🍓")


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

    col_info, col_btn = st.columns([3, 1])
    col_info.caption(f"{len(files)}개 이미지")

    # ZIP 생성 후 다운로드
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in files:
            path = os.path.join(base_dir, _LABEL_DIR_MAP[selected], fname)
            zf.write(path, arcname=fname)
    zip_buf.seek(0)

    col_btn.download_button(
        label="⬇ ZIP 다운로드",
        data=zip_buf,
        file_name=f"{_LABEL_DIR_MAP[selected]}.zip",
        mime="application/zip",
        use_container_width=True,
    )

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
        if st.session_state.get("is_developer"):
            st.divider()
            st.caption("🔐 개발자 모드 활성화됨")
            if st.button("로그아웃", use_container_width=True):
                st.session_state.is_developer = False
                st.rerun()
        else:
            with st.expander("···"):
                pw = st.text_input("비밀번호", type="password",
                                   key="dev_pw_input", label_visibility="collapsed")
                if st.button("확인", use_container_width=True):
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
        _render_image_result(uploaded_file, analysis_result)
    elif "video" in file_type:
        _render_video_result(analysis_result)

    _render_history_save_ui()

    st.divider()
    if st.button("🏠 처음 화면으로 돌아가기", use_container_width=True, type="primary"):
        temp_output = analysis_result.temp_output
        if temp_output and os.path.exists(temp_output):
            os.remove(temp_output)
        video_path = st.session_state.get("video_path")
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        st.session_state.uploaded_file = None
        st.session_state.video_path = None
        st.session_state.analysis_result = None
        st.session_state.analysis_type = None
        go_to("home")


def _render_image_result(uploaded_file, analysis_result):
    detection_result = analysis_result.result_list[0]


    # ------- 진단 결과 배너 -------
    st.title("📋 진단 결과")

    if detection_result.detection:
        info = disease_info[detection_result.class_id]
        st.markdown(f"""
<div style='background:#fff0f0; border:3px solid #FF4B4B; border-radius:14px;
     padding:1.2rem 1.5rem; margin-bottom:1rem;'>
  <p style='font-size:1.4rem; font-weight:bold; color:#cc0000; margin:0;'>
    🚨 {info['name']} 이(가) 발견되었습니다
  </p>
  <p style='font-size:1rem; color:#666; margin:0.3rem 0 0 0;'>
    AI 확신도: {detection_result.conf:.0%}
  </p>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div style='background:#f0fff4; border:3px solid #21c55d; border-radius:14px;
     padding:1.2rem 1.5rem; margin-bottom:1rem;'>
  <p style='font-size:1.4rem; font-weight:bold; color:#166534; margin:0;'>
    ✅ 건강한 딸기입니다!
  </p>
  <p style='font-size:1rem; color:#555; margin:0.3rem 0 0 0;'>
    병해충이 발견되지 않았습니다. 지금처럼 잘 관리해 주세요.
  </p>
</div>
""", unsafe_allow_html=True)

    # ------- 원본 vs 탐지 이미지 -------
    st.markdown("#### 🔍 사진 비교")
    col_orig, col_det = st.columns(2)
    with col_orig:
        st.caption("📷 올린 사진")
        uploaded_file.seek(0)
        st.image(uploaded_file.read(), use_container_width=True)
    with col_det:
        st.caption("🤖 AI가 표시한 부분")
        st.image(detection_result.annotated_frame, channels="BGR", use_container_width=True)

    # ------- 신뢰도 게이지 + 낮은 신뢰도 재촬영 안내 -------
    LOW_CONF_THRESHOLD = 0.5
    if detection_result.detection:
        st.markdown(f"**AI 확신도: {detection_result.conf:.0%}** (높을수록 더 확실합니다)")
        st.progress(detection_result.conf)
        if detection_result.conf < LOW_CONF_THRESHOLD:
            st.markdown("""
<div style='background:#fff3cd; border:2px solid #ffc107; border-radius:12px;
     padding:1rem 1.2rem; margin-top:0.8rem;'>
  <p style='font-size:1.1rem; font-weight:bold; color:#7d4e00; margin:0;'>
    📸 사진을 다시 찍어주세요
  </p>
  <p style='font-size:0.95rem; color:#5a3900; margin:0.4rem 0 0 0;'>
    AI가 잘 판단하지 못했습니다.<br>
    딸기를 <b>더 가까이, 밝은 곳에서</b> 다시 찍으면 더 정확하게 확인해 드립니다.
  </p>
</div>
""", unsafe_allow_html=True)
            if st.button("📸 다시 찍으러 가기", use_container_width=True):
                go_to("image")

    # ------- 병해 상세 정보 (접기) -------
    if detection_result.detection:
        st.divider()
        with st.expander("📖 이 병은 무엇인가요? (원인과 대처 방법 보기)  ▼ 탭하여 펼치기", expanded=False):
            utility.show_disease_info(detection_result.class_id)

    # ------- 피드백 & 학습 데이터 저장 -------
    st.divider()
    with st.container(border=True):
        st.markdown("### 💾 결과 저장하기")
        st.markdown("AI 결과가 맞으면 저장해 주세요. 앞으로 더 잘 진단하는 데 도움이 됩니다.")

        LABEL_OPTIONS = ["정상", "흰가루병", "잿빛곰팡이병"]
        class_to_label = {0: "잿빛곰팡이병", 1: "흰가루병"}
        default_label = class_to_label.get(detection_result.class_id, "정상") if detection_result.detection else "정상"

        selected_label = st.selectbox(
            "실제 상태를 선택해 주세요",
            LABEL_OPTIONS,
            index=LABEL_OPTIONS.index(default_label)
        )
        if st.button("💾 저장하기", use_container_width=True, type="primary", key="btn_training_save"):
            _save_training_image(uploaded_file, selected_label)

    if st.session_state.get("is_developer"):
        _render_developer_data_viewer()


def _render_video_result(analysis_result):
    analysis_type = st.session_state.analysis_type


    st.title("📋 진단 결과")
    detected = analysis_result.detection_frame_count

    # ------- 최상단 결과 배너 -------
    if detected == 0:
        st.markdown("""
<div style='background:#f0fff4; border:3px solid #21c55d; border-radius:14px;
     padding:1.2rem 1.5rem; margin-bottom:1rem;'>
  <p style='font-size:1.4rem; font-weight:bold; color:#166534; margin:0;'>
    ✅ 건강한 딸기입니다!
  </p>
  <p style='font-size:1rem; color:#555; margin:0.3rem 0 0 0;'>
    영상에서 병해충이 발견되지 않았습니다. 지금처럼 잘 관리해 주세요.
  </p>
</div>
""", unsafe_allow_html=True)
    else:
        names = "·".join([disease_info[cid]["name"] for cid in analysis_result.detected_classes])
        st.markdown(f"""
<div style='background:#fff0f0; border:3px solid #FF4B4B; border-radius:14px;
     padding:1.2rem 1.5rem; margin-bottom:1rem;'>
  <p style='font-size:1.4rem; font-weight:bold; color:#cc0000; margin:0;'>
    🚨 병해충이 발견되었습니다
  </p>
  <p style='font-size:1.05rem; color:#444; margin:0.4rem 0 0 0;'>
    발견된 병: <b>{names}</b>
  </p>
</div>
""", unsafe_allow_html=True)

    if analysis_type == "precise":
        col_vid, col_info = st.columns([3, 2])
        with col_vid:
            st.markdown("#### 🎬 분석된 영상")
            st.video(analysis_result.final_output)
            with open(analysis_result.final_output, "rb") as f:
                st.download_button(
                    label="⬇ 결과 영상 저장하기",
                    data=f,
                    file_name="result.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )
        with col_info:
            st.markdown("#### 📊 분석 요약")
            sorted_counts = _render_video_detection_summary(analysis_result, compact=True)

        if sorted_counts:
            st.divider()
            with st.expander("📖 이 병은 무엇인가요? (원인과 대처 방법 보기)  ▼ 탭하여 펼치기", expanded=False):
                for class_id, _ in sorted_counts:
                    utility.show_disease_info(class_id)

    elif analysis_type == "fast":
        _render_video_detection_summary(analysis_result)

        detected_frames = [r for r in analysis_result.result_list if r.detection]
        total = len(analysis_result.result_list)

        with st.expander(f"🖼 병해 발견된 장면 보기 ({len(detected_frames)}곳 발견)  ▼ 탭하여 펼치기", expanded=False):
            if not detected_frames:
                st.info("탐지된 장면이 없습니다.")
            else:
                cols_per_row = 3
                for i in range(0, len(detected_frames), cols_per_row):
                    row = detected_frames[i:i + cols_per_row]
                    cols = st.columns(cols_per_row)
                    for col, r in zip(cols, row):
                        info = disease_info.get(r.class_id, {})
                        col.image(r.annotated_frame, channels="BGR", use_container_width=True)
                        col.caption(f"{info.get('name','?')}  확신도 {r.conf:.0%}")
  
      


_RISK_COLOR = {"high": "#FF4B4B", "medium": "#f5a623", "none": "#21c55d"}
_RISK_LABEL = {"high": "위험", "medium": "주의", "none": "정상"}


def _render_history_save_ui():
    """결과 페이지 하단 — 이력 저장 여부 질문 + 이름 입력."""
    st.divider()

    if st.session_state.get("_result_saved"):
        if st.session_state.get("_result_save_status") == "saved":
            user_name = st.session_state.get("user_name", "")
            st.markdown(f"""
<div style='background:#f0fff4; border:2px solid #21c55d; border-radius:12px;
     padding:1rem 1.2rem;'>
  <p style='font-size:1.2rem; font-weight:bold; color:#166534; margin:0;'>
    ✅ 저장이 완료되었습니다!
  </p>
  <p style='font-size:1rem; color:#444; margin:0.3rem 0 0 0;'>
    <b>{user_name}</b>님의 진단 이력에 저장되었습니다.<br>
    사이드바의 <b>📋 진단 이력 보기</b>에서 확인하실 수 있습니다.
  </p>
</div>
""", unsafe_allow_html=True)
        return

    with st.container(border=True):
        st.markdown("### 📋 진단 이력 저장하기")
        st.markdown("이 진단 결과를 저장해 두시겠습니까?<br>이름을 입력하면 나중에 다시 확인할 수 있습니다.",
                    unsafe_allow_html=True)

        name = st.text_input(
            "이름",
            value=st.session_state.get("user_name", ""),
            placeholder="예: 홍길동 / 행복농장",
            label_visibility="collapsed",
            key="history_save_name",
        )

        col_save, col_skip = st.columns(2)
        with col_save:
            if st.button("💾 저장하기", use_container_width=True, type="primary", key="btn_history_save"):
                if not name.strip():
                    st.warning("이름을 입력해 주세요.")
                else:
                    st.session_state.user_name = name.strip()
                    # 이미지 or 동영상 이력 생성
                    uploaded_file = st.session_state.get("uploaded_file")
                    analysis_result = st.session_state.get("analysis_result")
                    analysis_type = st.session_state.get("analysis_type")
                    file_type = getattr(uploaded_file, "type", "")

                    if "image" in file_type and analysis_result:
                        detection_result = analysis_result.result_list[0]
                        entry = hist.make_image_entry(
                            detection_result,
                            getattr(uploaded_file, "name", ""),
                            user_name=name.strip(),
                        )
                    elif analysis_result:
                        entry = hist.make_video_entry(
                            analysis_result,
                            analysis_type or "",
                            user_name=name.strip(),
                        )
                    else:
                        entry = None

                    if entry:
                        hist.save_entry(entry)
                    st.session_state._result_saved = True
                    st.session_state._result_save_status = "saved"
                    st.rerun()

        with col_skip:
            if st.button("건너뛰기", use_container_width=True):
                st.session_state._result_saved = True
                st.session_state._result_save_status = "skipped"
                st.rerun()


def page_history():
    st.title("📋 진단 이력")
    st.markdown("<p style='font-size:1.1rem; color:#444;'>지금까지 진단한 결과를 확인할 수 있습니다.</p>", unsafe_allow_html=True)

    user_name = st.session_state.get("user_name", "")
    entries = hist.load_all(user_name=user_name)
    st.caption(f"👤 {user_name}님의 진단 이력")

    if not entries:
        st.info("아직 진단 이력이 없습니다. 사진이나 동영상을 진단하면 여기에 기록됩니다.")
        if st.button("🏠 처음 화면으로", use_container_width=True, type="primary"):
            go_to("home")
        return

    # 요약 지표
    total = len(entries)
    detected_count = sum(1 for e in entries if e.get("detected"))
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 전체 진단 수", f"{total}회")
    col2.metric("🦠 병 발견 횟수", f"{detected_count}회")
    col3.metric("✅ 이상 없음", f"{total - detected_count}회")

    st.divider()

    for entry in entries:
        with st.container(border=True):
            col_date, col_result, col_type = st.columns([3, 3, 2])

            col_date.markdown(f"🗓 **{entry.get('date', '')}**")

            if entry["type"] == "image":
                detected = entry.get("detected", False)
                risk = entry.get("risk", "none")
                color = _RISK_COLOR.get(risk, "#999")
                label = _RISK_LABEL.get(risk, "")
                name = entry.get("disease_name", "정상")
                conf = entry.get("confidence")
                if detected:
                    col_result.markdown(
                        f"<span style='color:{color}; font-weight:bold;'>🚨 {name}</span>"
                        + (f" <span style='color:#888; font-size:0.85rem;'>확신도 {conf:.0%}</span>" if conf else ""),
                        unsafe_allow_html=True
                    )
                else:
                    col_result.markdown("<span style='color:#21c55d; font-weight:bold;'>✅ 정상</span>", unsafe_allow_html=True)
                col_type.caption("📷 사진 진단")

            else:  # video
                detected = entry.get("detected", False)
                names = entry.get("disease_names", [])
                count = entry.get("detection_frame_count", 0)
                atype = "꼼꼼히" if entry.get("analysis_type") == "precise" else "빠른"
                if detected:
                    col_result.markdown(
                        f"<span style='color:#FF4B4B; font-weight:bold;'>🚨 {', '.join(names)}</span>"
                        f" <span style='color:#888; font-size:0.85rem;'>({count}장 발견)</span>",
                        unsafe_allow_html=True
                    )
                else:
                    col_result.markdown("<span style='color:#21c55d; font-weight:bold;'>✅ 정상</span>", unsafe_allow_html=True)
                col_type.caption(f"🎥 {atype} 분석")

    st.divider()
    if st.button("🏠 처음 화면으로", use_container_width=True, type="primary"):
        go_to("home")


def go_to(page):
    st.session_state.page = page
    st.rerun()


def go_to_top(page):
    """빈 loading 페이지를 경유해 스크롤을 최상단으로 초기화한 뒤 이동."""
    st.session_state.next_page = page
    st.session_state.page = "loading"
    st.rerun()
