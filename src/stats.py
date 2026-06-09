"""접속 횟수 및 분석 사용 횟수를 기록하고 조회하는 모듈."""
import json
import os
from zoneinfo import ZoneInfo
from datetime import datetime

_STATS_FILE = "history/stats.json"
_KST = ZoneInfo("Asia/Seoul")


def _load() -> dict:
    if not os.path.exists(_STATS_FILE):
        return {"access": 0, "analysis": {"image": 0, "fast": 0, "precise": 0}, "daily": {}}
    try:
        with open(_STATS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"access": 0, "analysis": {"image": 0, "fast": 0, "precise": 0}, "daily": {}}


def _save(data: dict) -> None:
    os.makedirs("history", exist_ok=True)
    with open(_STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def record_access() -> None:
    """새 세션 접속 시 호출."""
    data = _load()
    today = datetime.now(_KST).strftime("%Y-%m-%d")
    data["access"] = data.get("access", 0) + 1
    data.setdefault("daily", {}).setdefault(today, {"access": 0, "analysis": 0})
    data["daily"][today]["access"] += 1
    _save(data)


def record_analysis(analysis_type: str) -> None:
    """분석 완료 시 호출. analysis_type: 'image' | 'fast' | 'precise'"""
    data = _load()
    today = datetime.now(_KST).strftime("%Y-%m-%d")
    data.setdefault("analysis", {}).setdefault(analysis_type, 0)
    data["analysis"][analysis_type] += 1
    data.setdefault("daily", {}).setdefault(today, {"access": 0, "analysis": 0})
    data["daily"][today]["analysis"] += 1
    _save(data)


def get_stats() -> dict:
    return _load()
