from src.components import router, init_session_state, render_footer, inject_responsive_css
from streamlit_autorefresh import st_autorefresh

def main():

    # 10분마다 자동 새로고침 — 사용자가 페이지를 열어둔 동안 연결 유지
    st_autorefresh(interval=10 * 60 * 1000, key="keep_alive")

    init_session_state()
    inject_responsive_css()
    router()
    render_footer()


if __name__ == "__main__":
  main()
