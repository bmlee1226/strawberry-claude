from src.components import router, init_session_state, render_footer

def main():

  # -----------------------------------
  # session_state 초기화
  # -----------------------------------
  
  init_session_state()
  
  # -----------------------------------
  # router 실행
  # -----------------------------------
  
  router()
  
  render_footer()


if __name__ == "__main__":
  main()
