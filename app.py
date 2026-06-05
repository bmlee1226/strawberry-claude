from src.components import router, init_session_state, render_footer, inject_responsive_css

def main():

    init_session_state()
    inject_responsive_css()
    router()
    render_footer()


if __name__ == "__main__":
  main()
