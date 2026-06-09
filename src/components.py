import streamlit as st
import streamlit.components.v1 as components
from src import pages


def inject_responsive_css():
    st.markdown("""
<style>
/* ============================================================
   PC 기본 레이아웃
   ============================================================ */
.block-container {
    padding: 2rem 3rem !important;
    max-width: 1100px !important;
}

/* ============================================================
   개발자 로그인 — 사이드바 expander 거의 안 보이게
   ============================================================ */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    margin-bottom: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    background: transparent !important;
    color: #ccc !important;
    font-size: 0.72rem !important;
    font-weight: 400 !important;
    padding: 0.3rem 0.5rem !important;
    letter-spacing: 0.1em !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
    color: #999 !important;
    background: transparent !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"][open] summary {
    color: #999 !important;
    background: transparent !important;
    border-bottom: none !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] svg {
    display: none !important;
}

/* ============================================================
   Expander 강조 스타일 (메인 콘텐츠 전용)
   ============================================================ */

/* 닫힌 상태 */
[data-testid="stExpander"] {
    border: 2px solid #c8cdd6 !important;
    border-radius: 10px !important;
    margin-bottom: 0.8rem !important;
    overflow: hidden !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06) !important;
    transition: box-shadow 0.2s !important;
}

/* 헤더 영역 */
[data-testid="stExpander"] summary {
    background: linear-gradient(90deg, #f0f2f6, #f7f8fa) !important;
    padding: 0.85rem 1.1rem !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    color: #222 !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    user-select: none !important;
    letter-spacing: -0.01em !important;
}

/* 화살표 아이콘 크게 */
[data-testid="stExpander"] summary svg {
    width: 20px !important;
    height: 20px !important;
    color: #FF4B4B !important;
}

/* 호버 */
[data-testid="stExpander"]:not([open]):hover {
    border-color: #FF4B4B !important;
    box-shadow: 0 3px 10px rgba(255,75,75,0.15) !important;
}
[data-testid="stExpander"] summary:hover {
    background: linear-gradient(90deg, #ffe8e8, #fff3f3) !important;
    color: #FF4B4B !important;
}

/* 열린 상태 */
[data-testid="stExpander"][open] {
    border-color: #FF4B4B !important;
}
[data-testid="stExpander"][open] summary {
    border-bottom: 1.5px solid #ffd0d0 !important;
    border-radius: 8px 8px 0 0 !important;
    color: #FF4B4B !important;
    background: linear-gradient(90deg, #fff0f0, #fff7f7) !important;
}

/* 내부 콘텐츠 패딩 */
[data-testid="stExpander"] > div:last-child {
    padding: 0.8rem 1.1rem !important;
}

/* ============================================================
   고령 친화 — 큰 글씨·넓은 버튼
   ============================================================ */
.senior-step {
    background: #fff;
    border: 2px solid #e0e0e0;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    font-size: 1.1rem;
    line-height: 1.8;
}
.senior-step-num {
    display: inline-block;
    background: #FF4B4B;
    color: white;
    border-radius: 50%;
    width: 2rem;
    height: 2rem;
    text-align: center;
    line-height: 2rem;
    font-weight: bold;
    margin-right: 0.5rem;
    font-size: 1rem;
}
.senior-tip {
    background: #fffbe6;
    border-left: 4px solid #f5c518;
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem;
    font-size: 1rem;
    margin-top: 0.8rem;
    color: #555;
}

/* ============================================================
   모바일 (768px 이하)
   ============================================================ */
@media (max-width: 768px) {

  /* 여백 축소 */
  .block-container {
    padding: 1rem 0.75rem !important;
  }

  /* 버튼 — 터치하기 쉽게 크게 */
  .stButton > button {
    height: 3.2rem !important;
    font-size: 1rem !important;
    border-radius: 10px !important;
  }

  /* 제목 크기 조정 */
  h1 { font-size: 1.6rem !important; }
  h2 { font-size: 1.3rem !important; }
  h3 { font-size: 1.1rem !important; }

  /* metric 카드 패딩 축소 */
  [data-testid="metric-container"] {
    padding: 0.4rem !important;
  }
  [data-testid="metric-container"] label {
    font-size: 0.75rem !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.1rem !important;
  }

  /* 단계 표시바 — 모바일에서 줄바꿈 허용 */
  .step-bar {
    flex-wrap: wrap !important;
    gap: 0.3rem !important;
  }

  /* 사이드바 — 모바일에서 숨김 처리 후 햄버거로 접근 */
  [data-testid="stSidebar"] {
    min-width: 220px !important;
  }

  /* 파일 업로더 영역 */
  [data-testid="stFileUploader"] {
    padding: 0.5rem !important;
  }

  /* expander 헤더 */
  [data-testid="stExpander"] summary {
    font-size: 0.95rem !important;
  }

  /* progress bar 텍스트 */
  [data-testid="stProgressBar"] p {
    font-size: 0.8rem !important;
  }

  /* 예시 이미지 높이 축소 */
  .example-img-wrap img {
    height: 140px !important;
  }

  /* 컨테이너 테두리 카드 패딩 축소 */
  [data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 0.6rem !important;
  }
}

/* ============================================================
   태블릿 (769px ~ 1024px)
   ============================================================ */
@media (min-width: 769px) and (max-width: 1024px) {
  .block-container {
    padding: 1.5rem 1.5rem !important;
  }
  .example-img-wrap img {
    height: 180px !important;
  }
}

/* ============================================================
   사이드바 노인 친화 스타일
   ============================================================ */
[data-testid="stSidebar"] {
    background: #fafafa !important;
}
[data-testid="stSidebar"] h3 {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: #222 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    font-size: 1rem !important;
    color: #444 !important;
    line-height: 1.6 !important;
}
[data-testid="stSidebar"] .stButton > button {
    height: 3.2rem !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

def render_scroll_notice():
    """모바일에서만 표시되는 최상단 이동 안내."""
    st.markdown("""
<div class="scroll-notice">
  📌 페이지 이동 후 내용이 아래에 있다면 <b>위로 스크롤</b>해주세요.
</div>
<style>
.scroll-notice {
    display: none;
    background: #f0f2f6;
    border-radius: 8px;
    padding: 0.5rem 0.8rem;
    font-size: 0.82rem;
    color: #555;
    margin-bottom: 0.8rem;
}
@media (max-width: 768px) {
    .scroll-notice { display: block; }
}
</style>
""", unsafe_allow_html=True)


HOME_PAGE = "home"
IMAGE_PAGE = "image"
VIDEO_PAGE = "video"
REALTIME_VIDEO_PAGE = "realtime_video"
ANALYSIS_PAGE = "analysis"
RESULT_PAGE = "result"
LOADING_PAGE = "loading"
HISTORY_PAGE = "history"
NAME_PAGE = "name_input"


def router():


    render_scroll_notice()

    page = st.session_state.page

    if page == NAME_PAGE:
        pages.page_name_input()
        return

    if page == HOME_PAGE:
        pages.page_home()

    elif page == IMAGE_PAGE:
        pages.page_image()

    elif page == VIDEO_PAGE:
        pages.page_video()

    elif page == REALTIME_VIDEO_PAGE:
        pages.page_realtime_video()

    elif page == ANALYSIS_PAGE:
        pages.page_analysis()

    elif page == RESULT_PAGE:
        pages.page_result()

    elif page == HISTORY_PAGE:
        pages.page_history()

    elif page == LOADING_PAGE:
        # 빈 페이지 — 스크롤 위치를 0으로 초기화한 뒤 목적지로 이동
        with st.spinner(""):
            import time as _t
            _t.sleep(0.05)
        next_page = st.session_state.get("next_page", HOME_PAGE)
        st.session_state.page = next_page
        st.rerun()

    else:
        st.error("존재하지 않는 페이지입니다.")


def init_session_state():

    defaults = {
        "page": HOME_PAGE,
        "uploaded_file": None,
        "video_path": None,
        "analysis_type": None,
        "analysis_result": None,
        "conf_threshold": 0.3,
        "is_developer": False,
        "user_name": "",
        "weather_location": "",
        "weather_data": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # 세션당 1회 접속 카운트
    if not st.session_state.get("_access_counted"):
        from src import stats as _stats
        _stats.record_access()
        st.session_state._access_counted = True


def render_footer():

    # 사이드바 — 홈 버튼 + 개발자 모드
    with st.sidebar:
        # 이름 표시 + 변경
        user_name = st.session_state.get("user_name", "")
        if user_name:
            st.markdown(f"👤 **{user_name}**님")
            if st.button("이름 변경", use_container_width=True):
                st.session_state.user_name = ""
                st.session_state.page = NAME_PAGE
                st.rerun()
            st.divider()

        if st.session_state.get("page") != HISTORY_PAGE:
            if st.button("📋 진단 이력 보기", use_container_width=True):
                import os, tempfile
                st.session_state.page = HISTORY_PAGE
                st.rerun()

        if st.session_state.get("page") != HOME_PAGE:
            if st.button("🏠 처음 화면으로", use_container_width=True, type="primary"):
                # 임시 파일 정리
                analysis_result = st.session_state.get("analysis_result")
                if analysis_result:
                    temp_output = getattr(analysis_result, "temp_output", None)
                    if temp_output:
                        import os
                        if os.path.exists(temp_output):
                            os.remove(temp_output)
                video_path = st.session_state.get("video_path")
                if video_path:
                    import os
                    if os.path.exists(video_path):
                        os.remove(video_path)
                st.session_state.uploaded_file = None
                st.session_state.video_path = None
                st.session_state.analysis_result = None
                st.session_state.analysis_type = None
                st.session_state.page = HOME_PAGE
                st.rerun()
        st.divider()

    pages._render_developer_sidebar()

    st.markdown("---")
    st.caption("YOLO 기반 딸기 병해충 진단 시스템")
