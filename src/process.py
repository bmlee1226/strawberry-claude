import streamlit as st
from PIL import Image
import cv2

import tempfile
import subprocess
import time
import os

from src import utility
from src.disease_data import disease_info
from src.data_models import AnalysisResult

def process_image(uploaded_file, model, conf_threshold):
  image = Image.open(uploaded_file)
  
  st.divider()
  
  
  results = model(image, conf=conf_threshold)

  detection_result = utility.parse_detection_result(results)

  result_list = [detection_result]

  return AnalysisResult(result_list=result_list)

def process_fast_video(video_path, model, conf_threshold):
  
  videoinfo = utility.get_video_info(video_path)
  
  cap = cv2.VideoCapture(video_path)
  
  frame_count = 0

  detection_frame_count = 0

  detected_classes = set()
  detected_class_counts = {}
  
  progress_bar = st.progress(0)

  result_list = []
  
  while True:
      ret, frame = cap.read()
  
      if not ret:
          break
  
      # 진행률 표시
      progress = min(frame_count / videoinfo.total_frames, 1.0)
  
      progress_bar.progress(int(progress * 100))
  
  
      # 1초마다 1프레임 저장
      if frame_count % int(videoinfo.fps) == 0:
  
          results = model(frame, conf=conf_threshold)

          detection_result = utility.parse_detection_result(results)
        
          result_list.append(detection_result)
        
          if detection_result.detection:
              detection_frame_count += 1
              detected_classes.add(detection_result.class_id)
              cid = detection_result.class_id
              detected_class_counts[cid] = detected_class_counts.get(cid, 0) + 1

      frame_count += 1

  progress_bar.empty()
  cap.release()

  return AnalysisResult(
          result_list=result_list,
          detection_frame_count=detection_frame_count,
          detected_classes=detected_classes,
          detected_class_counts=detected_class_counts,
          conf_threshold=conf_threshold)


def process_precise_video(video_path, model, conf_threshold):
  
  videoinfo = utility.get_video_info(video_path)
  
  cap = cv2.VideoCapture(video_path)
  
  # -----------------------------
  # 결과 영상 저장 경로
  # -----------------------------
  
  temp_output = tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".mp4"
).name
  
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  
  out = cv2.VideoWriter(
      temp_output,
      fourcc,
      videoinfo.fps,
      (videoinfo.width, videoinfo.height)
  )
  
  # -----------------------------
  # 진행률 표시
  # -----------------------------
  progress_bar = st.progress(0)
  
  start_time = time.time()
  
  status_text = st.empty()
  
  preview_frame = st.empty()
  
  frame_idx = 0

  detection_frame_count = 0

  detected_classes = set()
  detected_class_counts = {}
  
  
  # -----------------------------
  # 프레임 처리
  # -----------------------------
  while cap.isOpened():
      ret, frame = cap.read()
  
      if not ret:
          break
  
      # YOLO 추론
      results = model(frame, conf=conf_threshold)
      detection_result = utility.parse_detection_result(results)

      if detection_result.detection:
        detected_classes.add(detection_result.class_id)
        detection_frame_count += 1
        cid = detection_result.class_id
        detected_class_counts[cid] = detected_class_counts.get(cid, 0) + 1
  
      # 저장
      out.write(detection_result.annotated_frame)
  
      frame_idx += 1
  
      progress = frame_idx / videoinfo.total_frames
      progress_bar.progress(progress)
  
      # -----------------------------
      # 시간 계산
      # -----------------------------
      elapsed_time = time.time() - start_time
  
      fps_processing = frame_idx / max(elapsed_time, 0.001)
  
      remaining_frames = videoinfo.total_frames - frame_idx
  
      remaining_time = remaining_frames / fps_processing
  
      # -----------------------------
      # 상태 표시
      # -----------------------------

      if frame_idx % 30 == 0:
        status_text.text(
            f"""
            처리 프레임: {frame_idx}/{videoinfo.total_frames}
            처리 FPS: {fps_processing:.2f}
            경과 시간: {elapsed_time:.1f}초
            남은 예상 시간: {remaining_time:.1f}초
            """
        )
    
      if frame_idx % 60 == 0:
      
          preview_frame.image(
              detection_result.annotated_frame,
              channels="BGR"
          )
        
  # 종료
  cap.release()
  out.release()

  st.success("분석 완료!")
  # -----------------------------
  # H.264 변환
  # -----------------------------
  final_output = tempfile.NamedTemporaryFile(
      delete=False,
      suffix=".mp4"
  ).name
  
  command = [
      "ffmpeg",
      "-y",
      "-i",
      temp_output,
      "-vcodec",
      "libx264",
      "-acodec",
      "aac",
      final_output
  ]
  
  try:
      subprocess.run(
          command,
          check=True
      )
  
  except Exception as e:
  
      st.error(f"영상 변환 실패: {e}")
  
  st.success("영상 생성 완료!")    
  os.remove(temp_output)

  return AnalysisResult(
          detection_frame_count=detection_frame_count,
          detected_classes=detected_classes,
          detected_class_counts=detected_class_counts,
          temp_output=temp_output,
          conf_threshold=conf_threshold,
          final_output=final_output)
  
  
