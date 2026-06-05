import requests

_DESC_KO = {
    "Sunny": "맑음",
    "Clear": "맑음",
    "Partly cloudy": "구름 조금",
    "Partly Cloudy": "구름 조금",
    "Cloudy": "흐림",
    "Overcast": "흐린",
    "Mist": "안개",
    "Fog": "짙은 안개",
    "Freezing fog": "결빙 안개",
    "Light drizzle": "가벼운 이슬비",
    "Freezing drizzle": "얼어붙는 이슬비",
    "Heavy freezing drizzle": "강한 얼어붙는 이슬비",
    "Light rain": "가벼운 비",
    "Moderate rain": "보통 비",
    "Heavy rain": "강한 비",
    "Light freezing rain": "가벼운 얼어붙는 비",
    "Moderate or heavy freezing rain": "강한 얼어붙는 비",
    "Light sleet": "가벼운 진눈깨비",
    "Moderate or heavy sleet": "강한 진눈깨비",
    "Light snow": "가벼운 눈",
    "Moderate snow": "보통 눈",
    "Heavy snow": "폭설",
    "Ice pellets": "우박",
    "Light rain shower": "가벼운 소나기",
    "Moderate or heavy rain shower": "강한 소나기",
    "Torrential rain shower": "폭우",
    "Light sleet showers": "가벼운 진눈깨비 소나기",
    "Moderate or heavy sleet showers": "강한 진눈깨비 소나기",
    "Light snow showers": "가벼운 눈 소나기",
    "Moderate or heavy snow showers": "강한 눈 소나기",
    "Light showers of ice pellets": "가벼운 우박",
    "Moderate or heavy showers of ice pellets": "강한 우박",
    "Patchy rain possible": "간헐적 비 가능",
    "Patchy snow possible": "간헐적 눈 가능",
    "Blowing snow": "날리는 눈",
    "Blizzard": "눈보라",
    "Thundery outbreaks possible": "뇌우 가능",
    "Patchy light drizzle": "간헐적 가벼운 이슬비",
    "Patchy light rain": "간헐적 가벼운 비",
    "Moderate rain at times": "때때로 보통 비",
    "Heavy rain at times": "때때로 강한 비",
    "Light rain with thunder": "천둥 동반 가벼운 비",
    "Moderate or heavy rain with thunder": "천둥 동반 강한 비",
    "Light snow with thunder": "천둥 동반 가벼운 눈",
    "Moderate or heavy snow with thunder": "천둥 동반 강한 눈",
}


def _translate_desc(desc: str) -> str:
    return _DESC_KO.get(desc, desc)


_RISK_CONFIG = {
    "high":   {"icon": "🔴", "label": "위험",  "color": "#FF4B4B"},
    "medium": {"icon": "🟡", "label": "주의",  "color": "#f5a623"},
    "low":    {"icon": "🟢", "label": "낮음",  "color": "#21c55d"},
}


def get_weather(location: str) -> dict | None:
    """wttr.in API로 현재 날씨를 조회한다 (API 키 불필요)."""
    try:
        url = f"https://wttr.in/{location}?format=j1&lang=ko"
        resp = requests.get(url, timeout=6)
        if resp.status_code != 200:
            return None
        data = resp.json()
        current = data["current_condition"][0]
        desc_en = current["weatherDesc"][0]["value"]
        return {
            "temp_c":     int(current["temp_C"]),
            "humidity":   int(current["humidity"]),
            "description": _translate_desc(desc_en),
            "feels_like": int(current["FeelsLikeC"]),
        }
    except Exception:
        return None


def assess_disease_risk(temp_c: int, humidity: int) -> list:
    """온도·습도를 기반으로 병해별 발생 위험도를 반환한다."""
    risks = []

    # 잿빛곰팡이병: 고습 + 서늘~온화 (10~25°C, 습도 70% 이상)
    if humidity >= 85 and 10 <= temp_c <= 25:
        risks.append({
            "disease": "잿빛곰팡이병",
            "level": "high",
            "reason": f"습도 {humidity}%로 매우 높고 기온 {temp_c}°C — 발생 최적 조건입니다. 즉시 환기하세요.",
        })
    elif humidity >= 70 and 10 <= temp_c <= 25:
        risks.append({
            "disease": "잿빛곰팡이병",
            "level": "medium",
            "reason": f"습도 {humidity}%로 높습니다. 환기를 강화하고 병든 잎·과실을 제거하세요.",
        })
    elif humidity >= 60 and 10 <= temp_c <= 25:
        risks.append({
            "disease": "잿빛곰팡이병",
            "level": "low",
            "reason": f"습도 {humidity}%로 다소 높습니다. 환기 상태를 점검하세요.",
        })

    # 흰가루병: 건조 + 온화~따뜻 (15°C 이상, 습도 70% 미만)
    if humidity < 45 and temp_c >= 15:
        risks.append({
            "disease": "흰가루병",
            "level": "high",
            "reason": f"습도 {humidity}%로 매우 낮고 기온 {temp_c}°C — 발생하기 매우 쉬운 환경입니다.",
        })
    elif humidity < 60 and temp_c >= 15:
        risks.append({
            "disease": "흰가루병",
            "level": "medium",
            "reason": f"습도 {humidity}%로 낮습니다. 잎 뒷면을 자주 확인하세요.",
        })
    elif humidity < 70 and temp_c >= 20:
        risks.append({
            "disease": "흰가루병",
            "level": "low",
            "reason": f"기온 {temp_c}°C로 높아 흰가루병에 주의하세요.",
        })

    return risks


def risk_config(level: str) -> dict:
    return _RISK_CONFIG.get(level, {"icon": "⚪", "label": "정보 없음", "color": "#999"})
