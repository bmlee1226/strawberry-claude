import streamlit as st
from src import pages

HOME_PAGE = "home"
IMAGE_PAGE = "image"
VIDEO_PAGE = "video"
REALTIME_VIDEO_PAGE = "realtime_video"
ANALYSIS_PAGE = "analysis"
RESULT_PAGE = "result"


def router():

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
        "conf_threshold": 0.3
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


def render_footer():

    st.markdown("---")

    st.caption(
        "YOLO 기반 딸기 병해충 진단 시스템"
    )
