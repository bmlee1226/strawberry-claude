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
</style>
""", unsafe_allow_html=True)

HOME_PAGE = "home"
IMAGE_PAGE = "image"
VIDEO_PAGE = "video"
REALTIME_VIDEO_PAGE = "realtime_video"
ANALYSIS_PAGE = "analysis"
RESULT_PAGE = "result"


def router():

    # 페이지 이동 시 최상단으로 스크롤
    components.html("<script>window.parent.scrollTo({top: 0, behavior: 'instant'});</script>", height=0)

    page = st.session_state.page

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
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


def render_footer():

    pages._render_developer_sidebar()

    st.markdown("---")

    st.caption(
        "YOLO 기반 딸기 병해충 진단 시스템"
    )
