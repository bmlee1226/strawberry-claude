import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

_KST = ZoneInfo("Asia/Seoul")


def _now_kst() -> datetime:
    return datetime.now(_KST)

_HISTORY_FILE = "history/history.json"
_MAX_ENTRIES = 100


def save_entry(entry: dict) -> None:
    os.makedirs("history", exist_ok=True)
    history = load_all()
    history.insert(0, entry)
    history = history[:_MAX_ENTRIES]
    with open(_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def load_all(user_name: str = "") -> list:
    if not user_name or not os.path.exists(_HISTORY_FILE):
        return []
    try:
        with open(_HISTORY_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return [e for e in data if e.get("user_name") == user_name]
    except Exception:
        return []


def make_image_entry(detection_result, file_name: str = "", user_name: str = "") -> dict:
    from src.disease_data import disease_info
    if detection_result.detection:
        info = disease_info.get(detection_result.class_id, {})
        disease_name = info.get("name", "알 수 없음")
        risk = info.get("risk", "none")
    else:
        disease_name = "정상"
        risk = "none"

    return {
        "id": _now_kst().strftime("%Y%m%d_%H%M%S"),
        "date": _now_kst().strftime("%Y년 %m월 %d일 %H:%M"),
        "type": "image",
        "user_name": user_name,
        "detected": detection_result.detection,
        "disease_name": disease_name,
        "risk": risk,
        "confidence": round(detection_result.conf, 3) if detection_result.conf else None,
        "file_name": file_name,
    }


def make_video_entry(analysis_result, analysis_type: str, user_name: str = "") -> dict:
    from src.disease_data import disease_info
    detected_names = [
        disease_info.get(cid, {}).get("name", "알 수 없음")
        for cid in analysis_result.detected_classes
    ]
    return {
        "id": _now_kst().strftime("%Y%m%d_%H%M%S"),
        "date": _now_kst().strftime("%Y년 %m월 %d일 %H:%M"),
        "type": "video",
        "user_name": user_name,
        "analysis_type": analysis_type,
        "detected": analysis_result.detection_frame_count > 0,
        "disease_names": detected_names,
        "detection_frame_count": analysis_result.detection_frame_count,
    }
