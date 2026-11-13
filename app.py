import datetime as dt
import os
import time
from pathlib import Path

import folium
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium


# ==========================================
# 0. 경로 및 상수
# ==========================================
APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
DATA_DIR = APP_DIR / "data"

SLICE_FILE = "app_slice.parquet"          # 앱용 소형 데이터 (feature 테이블의 일부)
STATIONS_FILE = "selected_stations_60.csv"  # 대여소 이름/구 정보

TARGET_RENTALS = "rentals_target_1h"
TARGET_RETURNS = "returns_target_1h"

MODEL_OPTIONS = ["GRU", "LightGBM", "XGBoost"]
PERIOD_OPTIONS = {"오전 출근": "morning", "오후 퇴근": "evening"}  # commute_period 값과 매칭
WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]
MAX_DAYS_AHEAD = 365  # 날짜 선택 가능 범위: 오늘 ~ 1년 뒤

# 자전거 경로: FOSSGIS가 운영하는 공개 OSRM 서버(자전거 프로필). 비상업·소규모 사용용이며 초당 1회 이하로 호출해야 한다.
ROUTING_BASE = "https://routing.openstreetmap.de/routed-bike"
ROUTING_USER_AGENT = "ddareungi-esg-demo/1.0"  # 서버 이용 규칙상 앱을 식별하는 값 (연락처를 덧붙이는 것을 권장)
ROUTING_TIMEOUT_S = 5
ROUTING_RETRY_AFTER_S = 300  # 경로 서버 호출이 실패하면 이 시간(초) 동안은 다시 부르지 않고 추정값을 쓴다
NEAREST_CANDIDATES = 3  # 출발지·도착지마다 보여 줄 가까운 대여소 후보 수

# 4번 메뉴(따릉이 길찾기): 주소 검색은 카카오 로컬 API
# 키 KAKAO_REST_KEY는 환경변수를 먼저 보고, 없으면 Streamlit secrets(.streamlit/secrets.toml 또는 Cloud Secrets)를 본다.
KAKAO_BASE = "https://dapi.kakao.com/v2/local/search"
KAKAO_TIMEOUT_S = 5
SEOUL_RECT = "126.76,37.41,127.19,37.72"  # 장소 이름 검색 범위: 서울을 감싸는 사각형 (경도,위도,경도,위도)
MAX_STATION_DIST_M = 2000   # 출발지·도착지에서 이 거리(직선)보다 먼 대여소는 후보에서 뺀다
WARN_STATION_DIST_M = 1000  # 이 거리보다 멀면 "대여소까지 멀다" 경고
BIKE_KMH_ESTIMATE = 15.0    # 경로 서버 실패 시 추정용 자전거 평균 속도 (가정값)
DETOUR_ESTIMATE = 1.3       # 경로 서버 실패 시 직선거리에 곱하는 우회 보정 (가정값)


# ==========================================
# 1. 페이지 및 기본 레이아웃 설정
# ==========================================
st.set_page_config(
    page_title="따릉이 출퇴근 수요 예측",
    page_icon="🚲",
    layout="wide",
)

st.title("🚲 AI 기반 따릉이 출퇴근 수요 예측 및 대여소 안내 서비스")
st.caption(
    "출퇴근 피크 시간대 주거지 ↔ 업무지 간 역방향 수요 예측, K-Means 클러스터 분석, "
    "시간대별 예측 그래프, 자전거 경로 기반 가까운 대여소 안내를 제공합니다."
)

# 사이드바: 접기(◀) 버튼을 '통합 운영 제어판' 제목과 같은 줄로 올리고, 내용을 위로 당긴다.
# (스트림릿 내부 구조(data-testid)에 의존하므로, 버전이 바뀌면 어긋날 수 있다)
st.markdown(
    """
    <style>
    [data-testid="stSidebarHeader"] {
        position: absolute !important;
        top: calc(0.75rem + 20px) !important;
        right: 0.75rem !important;
        height: 2.25rem !important;
        min-height: 0 !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        z-index: 2;
    }
    [data-testid="stSidebarUserContent"] {
        padding-top: calc(0.75rem + 20px) !important;
    }
    .stat-row {
        display: grid;
        grid-template-columns: repeat(3, 6.5rem);
        column-gap: 0.25rem;
        justify-content: end;
        align-items: start;
    }
    .stat-label { font-size: 0.75rem; opacity: 0.75; white-space: nowrap; }
    .stat-help {
        display: inline-block;
        width: 0.95rem;
        height: 0.95rem;
        line-height: 0.95rem;
        text-align: center;
        font-size: 0.65rem;
        border: 1px solid currentColor;
        border-radius: 50%;
        opacity: 0.6;
        cursor: help;
        margin-left: 0.2rem;
    }
    .stat-line { display: flex; align-items: center; gap: 0.5rem; white-space: nowrap; }
    .stat-value { font-size: 1.2rem; }
    .stat-pill { font-size: 0.7rem; padding: 0.1rem 0.5rem; border-radius: 1rem; }
    .stat-pill.down { color: #09ab3b; background: rgba(33, 195, 84, 0.12); }
    .stat-pill.up { color: #ff2b2b; background: rgba(255, 43, 43, 0.12); }
    .compact-heading {
        font-size: 1.2rem;
        font-weight: 700;
        line-height: 1.4;
        white-space: nowrap;
        margin: 0.25rem 0;
    }
    [data-testid="stSidebarUserContent"] hr {
        margin: 1rem 0 !important;
    }
    [data-testid="stSidebarUserContent"] h3 {
        padding-top: 0.25rem !important;
    }
    [data-testid="stSidebarUserContent"] h2 {
        font-size: 1.25rem !important;
        line-height: 2.25rem !important;
        margin: 0 !important;
        padding: 0 3rem 0 0 !important;
        white-space: nowrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 2. 파일 위치 확인 및 데이터/모델 로딩 (캐싱 처리)
# ==========================================
def find_data_file(name):
    """data/ 폴더를 먼저, 없으면 app.py와 같은 폴더에서 찾는다."""
    for folder in (DATA_DIR, APP_DIR):
        path = folder / name
        if path.exists():
            return path
    return DATA_DIR / name


def model_path(model_name, target):
    if model_name == "GRU":
        return MODEL_DIR / f"bicycle_gru_{target}.keras"
    key = "lightgbm" if model_name == "LightGBM" else "xgboost"
    return MODEL_DIR / f"bicycle_{key}_{target}.pkl"


def missing_files(model_name):
    meta_name = "gru_meta.pkl" if model_name == "GRU" else "feature_meta.pkl"
    needed = [
        find_data_file(SLICE_FILE),
        find_data_file(STATIONS_FILE),
        model_path(model_name, TARGET_RENTALS),
        model_path(model_name, TARGET_RETURNS),
        MODEL_DIR / meta_name,
    ]
    return [p for p in needed if not p.exists()]


@st.cache_data(show_spinner=False)
def load_slice():
    df = pd.read_parquet(find_data_file(SLICE_FILE))
    df["station_id"] = df["station_id"].astype(str)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    return df


def read_stations_csv():
    path = find_data_file(STATIONS_FILE)
    try:
        stations = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        stations = pd.read_csv(path, encoding="cp949")
    stations["station_id"] = stations["station_id"].astype(str)
    return stations


@st.cache_data(show_spinner=False)
def load_stations():
    return read_stations_csv()[["station_id", "station_name", "gu", "group"]]


@st.cache_data(show_spinner=False)
def load_station_locations():
    """5번 메뉴용: 대여소 이름·구·좌표 (모델/예측 데이터 없이 CSV만 사용)."""
    stations = read_stations_csv()[["station_id", "station_name", "gu", "lat", "lon"]]
    return stations.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def pick_reference_date(target_date, df):
    """선택한 날짜와 같은 요일인 '가장 최근' 실제 데이터 날짜를 고른다.
    평일은 휴일이 아닌 날을 우선한다. (공휴일 달력이 2025년까지라 선택일의 공휴일 여부는 반영하지 않음)"""
    dates = pd.DataFrame({"date": sorted(df["date"].unique())})
    dates["weekday"] = [d.weekday() for d in dates["date"]]
    if "is_offday" in df.columns:
        offday = df.groupby("date")["is_offday"].max().fillna(False).astype(bool)
        dates["offday"] = dates["date"].map(offday).astype(bool)
    else:
        dates["offday"] = False

    candidates = dates[dates["weekday"] == target_date.weekday()]
    if target_date.weekday() < 5:
        non_off = candidates[~candidates["offday"]]
        if not non_off.empty:
            candidates = non_off
    return None if candidates.empty else candidates["date"].max()


def sidebar_station_picker(disabled=False):
    """사이드바 대여소 드롭다운. 선택한 station_id를 돌려주고, '전체 대여소'면 None."""
    if not find_data_file(STATIONS_FILE).exists():
        return None
    stations = load_stations()
    names = stations["station_name"].fillna(stations["station_id"])
    duplicated = names.duplicated(keep=False)  # 이름이 같은 대여소는 ID를 덧붙여 구분
    labels = np.where(duplicated, names + " (" + stations["station_id"] + ")", names)
    lookup = dict(zip(labels, stations["station_id"]))
    choice = st.sidebar.selectbox(
        "📍 대여소 선택",
        ["전체 대여소"] + sorted(lookup),
        disabled=disabled,
        help="지도(1·2번 메뉴)는 선택한 대여소로 확대되고, 선 그래프(3번 메뉴)는 해당 대여소만 보여주며, "
             "4번 메뉴(길찾기)에서는 쓰지 않습니다.",
    )
    return None if choice == "전체 대여소" else lookup[choice]


@st.cache_resource(show_spinner="모델을 불러오는 중입니다...")
def load_model(model_name, target):
    path = model_path(model_name, target)
    if model_name == "GRU":
        from tensorflow import keras  # GRU 선택 시에만 불러와 시작 속도 확보

        return keras.models.load_model(path, compile=False)
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_meta(model_name):
    meta_name = "gru_meta.pkl" if model_name == "GRU" else "feature_meta.pkl"
    return joblib.load(MODEL_DIR / meta_name)


# ==========================================
# 3. 모델 입력 만들기 + 예측
#    (학습 스크립트와 동일한 전처리를 그대로 재현)
# ==========================================
def build_tree_features(df, meta):
    """LightGBM / XGBoost: 학습 때와 같은 category 목록으로 변환 (NaN은 그대로 둠)."""
    X = df[meta["inputs"]].copy()
    for col, cats in meta["categories"].items():
        if col in X.columns:
            X[col] = pd.Categorical(X[col], categories=cats)
    return X


def build_gru_inputs(df, meta):
    """GRU: bool→float, 숫자 결측 0 대체, 범주형은 학습 때 저장한 정수 코드로 변환."""
    num_cols, cat_cols, cat_maps = meta["num_cols"], meta["cat_cols"], meta["cat_maps"]
    X = df[num_cols + cat_cols].copy()

    for col in num_cols:
        if pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype("float64")
    for col in num_cols:
        X[col] = X[col].fillna(0.0).astype("float64")

    inputs = {}
    for col in cat_cols:
        s = X[col].astype(str).fillna("missing")
        unknown_code = len(cat_maps[col])  # Embedding(input_dim=num_classes + 1)의 여분 칸
        codes = s.map(cat_maps[col]).fillna(unknown_code).astype("int64")
        inputs[f"input_{col}"] = codes.values
    inputs["input_numeric"] = X[num_cols].values
    return inputs


def predict_rows(model_name, target, df):
    model = load_model(model_name, target)
    meta = load_meta(model_name)
    if model_name == "GRU":
        pred = model.predict(build_gru_inputs(df, meta), batch_size=4096, verbose=0)
        pred = np.asarray(pred).reshape(-1)
    else:
        pred = model.predict(build_tree_features(df, meta))
    return np.clip(np.asarray(pred, dtype=float), 0, None)  # 학습 평가와 동일하게 음수는 0으로


@st.cache_data(show_spinner="선택한 모델로 예측 중입니다...")
def run_prediction(model_name, date, period_label):
    """선택한 날짜·피크 시간대의 대여소별 대여/반납을 예측하고 Net-flow를 계산한다."""
    period = PERIOD_OPTIONS[period_label]
    df = load_slice()
    sub = df[(df["date"] == date) & (df["commute_period"].astype(str) == period)].copy()
    if sub.empty:
        return None, {}

    sub = sub.sort_values(["station_id", "timestamp"])
    sub["pred_rentals"] = predict_rows(model_name, TARGET_RENTALS, sub)
    sub["pred_returns"] = predict_rows(model_name, TARGET_RETURNS, sub)

    # 대여소별로 피크 시간대 전체(시간별 예측의 합)를 집계
    spec = {
        "위도": ("latitude", "first"),
        "경도": ("longitude", "first"),
        "총거치대수": ("dock_count", "first"),
        "현재자전거수": ("inventory_lag_1h", lambda s: s.iloc[0]),  # 시간대 시작 시점의 재고
        "예측_대여": ("pred_rentals", "sum"),
        "예측_반납": ("pred_returns", "sum"),
    }
    has_actual = {TARGET_RENTALS, TARGET_RETURNS}.issubset(sub.columns)
    if has_actual:
        spec["실제_대여"] = (TARGET_RENTALS, lambda s: s.sum(min_count=1))
        spec["실제_반납"] = (TARGET_RETURNS, lambda s: s.sum(min_count=1))
    out = sub.groupby("station_id").agg(**spec).reset_index()

    excluded = int(out[["현재자전거수", "총거치대수"]].isna().any(axis=1).sum())
    out = out.dropna(subset=["현재자전거수", "총거치대수"]).copy()

    out["순수요_NetFlow"] = out["예측_반납"] - out["예측_대여"]  # 반납 - 대여
    if has_actual:
        out["실제_NetFlow"] = out["실제_반납"] - out["실제_대여"]
    else:
        out["실제_NetFlow"] = np.nan

    out["예측_잔여수량"] = (
        (out["현재자전거수"] + out["순수요_NetFlow"])
        .round()
        .clip(lower=0, upper=out["총거치대수"])
        .astype(int)
    )

    conditions = [
        (out["예측_잔여수량"] <= 3),
        (out["예측_잔여수량"] >= out["총거치대수"] - 3),
    ]
    choices = ["부족 위험 🔴 (대여 불가)", "포화 위험 🔵 (반납 불가)"]
    out["위험상태"] = np.select(conditions, choices, default="정상 🟢")
    out["적용모델"] = model_name

    # 대여소 이름/구 정보 붙이기
    out = out.merge(load_stations(), on="station_id", how="left")
    unmatched = int(out["station_name"].isna().sum())
    out["대여소ID"] = out["station_id"]
    out["대여소명"] = out["station_name"].fillna(out["station_id"])
    out["클러스터"] = out["gu"].fillna("미상")  # 클러스터 = 자치구
    out["구분"] = out["group"].fillna("미상")
    out = out.drop(columns=["station_name", "gu", "group"])

    # 관측 기상 (모델 입력으로 쓰인 1시간 전 관측값; 서울 단일 관측소라 모든 대여소가 같음)
    wx = sub.groupby("timestamp").agg(
        temp=("temp_c_lag1h", "mean"), precip=("precip_mm_lag1h", "mean")
    )
    info = {
        "hour_min": int(sub["hour"].min()),
        "hour_max": int(sub["hour"].max()),
        "temp": float(wx["temp"].mean()),
        "precip": float(wx["precip"].sum(min_count=1)),
        "excluded": excluded,
        "unmatched": unmatched,
        "has_actual": has_actual,
    }
    return out, info


@st.cache_data(show_spinner="선택한 모델로 하루치 예측 중입니다...")
def run_daily_prediction(model_name, date):
    """참조일 하루(0~23시) 전체의 대여소별 시간당 대여/반납을 예측한다."""
    df = load_slice()
    sub = df[df["date"] == date].copy()
    if sub.empty:
        return None

    sub["pred_rentals"] = predict_rows(model_name, TARGET_RENTALS, sub)
    sub["pred_returns"] = predict_rows(model_name, TARGET_RETURNS, sub)

    sub = sub.merge(load_stations(), on="station_id", how="left")
    sub["대여소ID"] = sub["station_id"]
    sub["대여소명"] = sub["station_name"].fillna(sub["station_id"])
    return sub[["대여소ID", "대여소명", "hour", "pred_rentals", "pred_returns"]]


def haversine_m(lat1, lon1, lat2, lon2):
    """두 지점 사이의 직선거리(m). lat2/lon2는 배열이어도 된다."""
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    d_phi = p2 - p1
    d_lambda = np.radians(lon2) - np.radians(lon1)
    a = np.sin(d_phi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(d_lambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def nearest_to_point(stations, lat, lon, k=NEAREST_CANDIDATES, max_m=MAX_STATION_DIST_M):
    """임의 위치에서 직선거리로 가까운 대여소 k곳 (max_m 이내만). 없으면 빈 표."""
    out = stations.copy()
    out["straight_m"] = haversine_m(lat, lon, out["lat"].to_numpy(dtype=float), out["lon"].to_numpy(dtype=float))
    return out[out["straight_m"] <= max_m].nsmallest(k, "straight_m").reset_index(drop=True)


def one_based(df):
    """표의 행 번호(맨 왼쪽 열)가 0이 아니라 1부터 시작하도록 바꾼다."""
    out = df.reset_index(drop=True)
    out.index = out.index + 1
    return out


def fmt_distance(meters):
    return f"{meters / 1000:.1f} km" if meters >= 1000 else f"{meters:,.0f} m"


def osrm_get(path, params):
    resp = requests.get(
        f"{ROUTING_BASE}/{path}",
        params=params,
        headers={"User-Agent": ROUTING_USER_AGENT},
        timeout=ROUTING_TIMEOUT_S,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "Ok":
        raise ValueError(f"routing error: {data.get('code')} {data.get('message', '')}")
    return data


@st.cache_data(show_spinner="자전거 경로를 찾는 중입니다...", ttl=24 * 3600)
def bike_route(a_lat, a_lon, b_lat, b_lon):
    """대여소 → 대여소 자전거 경로. 실패하면 예외를 던진다(실패 결과가 캐시되지 않도록)."""
    try:
        data = osrm_get(
            f"route/v1/driving/{a_lon:.6f},{a_lat:.6f};{b_lon:.6f},{b_lat:.6f}",
            {"overview": "full", "geometries": "geojson"},
        )
    finally:
        time.sleep(1.1)  # 공개 서버 이용 규칙: 초당 1회 이하
    route = data["routes"][0]
    return {
        "distance_m": float(route["distance"]),
        "duration_s": float(route["duration"]),
        "estimated": False,
        "path": [[la, lo] for lo, la in route["geometry"]["coordinates"]],  # GeoJSON은 (경도, 위도) 순서
    }


def bike_leg(a, b):
    """경로 서버를 먼저 쓰고, 실패하면 직선거리×보정 ÷ 평균속도로 추정한다.
    실패한 뒤 ROUTING_RETRY_AFTER_S 동안은 서버를 다시 부르지 않아 화면이 느려지지 않게 한다."""
    if time.time() >= st.session_state.get("routing_down_until", 0):
        try:
            return bike_route(a[0], a[1], b[0], b[1])
        except Exception:
            st.session_state["routing_down_until"] = time.time() + ROUTING_RETRY_AFTER_S
    dist = float(haversine_m(a[0], a[1], b[0], b[1])) * DETOUR_ESTIMATE
    return {"distance_m": dist, "duration_s": dist / (BIKE_KMH_ESTIMATE * 1000 / 3600),
            "estimated": True, "path": [list(a), list(b)]}


def fit_view(points, width_px=600, height_px=480, padding_px=50, max_zoom=17):
    """점들이 모두 보이는 지도 중심과 확대 단계(웹 메르카토르 기준).
    width_px는 지도 실제 폭을 알 수 없어 좁은 화면 기준으로 잡은 값이다(넓으면 여유가 더 생길 뿐)."""
    lats = np.array([p[0] for p in points], dtype=float)
    lons = np.array([p[1] for p in points], dtype=float)
    merc = np.log(np.tan(np.pi / 4 + np.radians(lats) / 2))  # 위도 → 메르카토르 y
    dx = (lons.max() - lons.min()) / 360.0
    dy = (merc.max() - merc.min()) / (2 * np.pi)
    zooms = [max_zoom]
    if dx > 0:
        zooms.append(np.log2((width_px - 2 * padding_px) / (256 * dx)))
    if dy > 0:
        zooms.append(np.log2((height_px - 2 * padding_px) / (256 * dy)))
    zoom = int(np.clip(np.floor(min(zooms)), 3, max_zoom))
    center_lat = float(np.degrees(2 * np.arctan(np.exp((merc.max() + merc.min()) / 2)) - np.pi / 2))
    return [center_lat, float((lons.max() + lons.min()) / 2)], zoom


def kakao_key():
    """환경변수 → Streamlit secrets 순서로 카카오 REST API 키를 찾는다. 없으면 None."""
    key = os.environ.get("KAKAO_REST_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets["KAKAO_REST_KEY"]).strip() or None
    except Exception:  # secrets.toml이 없거나 키가 없음
        return None


@st.cache_data(show_spinner="주소를 찾는 중입니다...", ttl=24 * 3600)
def kakao_search(query):
    """장소 이름 → 주소 순으로 검색. 결과 [{name, address, lat, lon}] (최대 5개, 서울 주소 우선).
    장소 이름은 서울 범위(SEOUL_RECT) 안에서만 찾는다(예: '중앙시장'이 광주로 잡히지 않도록).
    오류는 requests.HTTPError 등 예외로 던진다(캐시되지 않도록)."""
    headers = {"Authorization": f"KakaoAK {kakao_key()}"}
    for kind, extra in (("keyword", {"rect": SEOUL_RECT}), ("address", {})):
        resp = requests.get(f"{KAKAO_BASE}/{kind}.json", headers=headers,
                            params={"query": query, "size": 5, **extra}, timeout=KAKAO_TIMEOUT_S)
        resp.raise_for_status()
        docs = resp.json().get("documents", [])
        if docs:
            docs = sorted(docs, key=lambda d: not d.get("address_name", "").startswith("서울"))  # 서울 먼저(순서 유지)
            return [{
                "name": d.get("place_name") or d.get("address_name", query),
                "address": d.get("road_address_name") or d.get("address_name", ""),
                "lat": float(d["y"]), "lon": float(d["x"]),  # 카카오는 x=경도, y=위도
            } for d in docs]
    return []


def station_forecast():
    """사이드바 조건(모델·날짜·피크 시간대)의 대여소별 예측 상태. (표 또는 None, 안내 문구).
    예측이 안 돼도 길찾기는 계속되도록 st.stop() 없이 None을 돌려준다."""
    if missing_files(model_type):
        return None, "예측 모델·데이터 파일이 없어 재고 예측 상태는 표시하지 않습니다."
    try:
        ref_date = pick_reference_date(selected_date, load_slice())
        if ref_date is None:
            return None, "참조할 같은 요일 데이터가 없어 재고 예측 상태는 표시하지 않습니다."
        df_pred, _ = run_prediction(model_type, ref_date, period_label)
    except Exception:
        return None, "예측 중 오류가 나서 재고 예측 상태는 표시하지 않습니다."
    if df_pred is None:
        return None, "참조일에 해당 시간대 데이터가 없어 재고 예측 상태는 표시하지 않습니다."
    note = (f"재고 상태는 {selected_date} {period_label} 예측(같은 요일 참조일 {ref_date} 데이터, {model_type}) 기준의 "
            "참고용 시뮬레이션입니다. 실제 실시간 재고가 아닙니다.")
    return df_pred.set_index("대여소ID")[["위험상태", "예측_잔여수량", "총거치대수"]], note


def hourly_totals(daily, station_id=None):
    """시간(0~23)별 예측 대여/반납 합계. station_id가 없으면 전체 대여소 합계."""
    sub = daily if station_id is None else daily[daily["대여소ID"] == station_id]
    hourly = sub.groupby("hour")[["pred_rentals", "pred_returns"]].sum().reindex(range(24))
    hourly.columns = ["예측 대여", "예측 반납"]
    hourly.index.name = "시간"
    return hourly


# ==========================================
# 4. 사이드바 제어판
# ==========================================
st.sidebar.header("🔍 통합 운영 제어판")
mode = st.sidebar.radio(
    "메뉴 선택",
    [
        "1. 실시간 수급 불균형 지도",
        "2. K-Means 클러스터 분석",
        "3. 시간대별 대여·반납 예측 (선 그래프)",
        "4. 가까운 대여소 찾기 (경로 지도)",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 예측 AI 모델 및 조건 설정")
model_type = st.sidebar.selectbox("예측 모델 선택", MODEL_OPTIONS)

today = pd.Timestamp.now(tz="Asia/Seoul").date()
selected_date = st.sidebar.date_input(
    "날짜 선택 (오늘 이후)",
    value=today,
    min_value=today,
    max_value=today + dt.timedelta(days=MAX_DAYS_AHEAD),
)
period_label = st.sidebar.selectbox("피크 시간대 선택", list(PERIOD_OPTIONS))
selected_station_id = sidebar_station_picker(disabled=mode.startswith("4."))
st.sidebar.caption(
    "선택한 날짜는 아직 실제 데이터가 없으므로, 같은 요일의 가장 최근 실제 데이터(대여·반납 이력, 재고, 기상)를 "
    "모델에 입력해 예측하는 시뮬레이션입니다."
)


def get_reference_date():
    """파일 확인 → 참조일 선택 → 시뮬레이션 안내 배너. 예측이 필요한 모드가 공통으로 사용."""
    problems = missing_files(model_type)
    if problems:
        st.error("예측에 필요한 파일을 찾을 수 없습니다. 아래 위치에 파일이 있는지 확인해 주세요.")
        st.code("\n".join(str(p.relative_to(APP_DIR)) if APP_DIR in p.parents else str(p) for p in problems))
        st.stop()

    ref_date = pick_reference_date(selected_date, load_slice())
    if ref_date is None:
        st.warning(
            "선택한 날짜와 같은 요일의 참조 데이터가 없습니다. "
            "app_slice.parquet에 최소 2주 이상의 데이터가 들어 있는지 확인해 주세요."
        )
        st.stop()

    st.info(
        f"🧪 **시뮬레이션 결과입니다.** 선택한 {selected_date}({WEEKDAYS[selected_date.weekday()]})은 "
        f"아직 실제 데이터가 없어, 같은 요일인 **{ref_date}**의 실제 데이터(대여·반납 이력, 재고, 기상)를 "
        f"{model_type} 모델에 입력해 예측했습니다. 실제 기상 예보가 아니며, 공휴일은 반영되지 않습니다."
    )
    return ref_date


def get_predictions():
    ref_date = get_reference_date()
    df_pred, info = run_prediction(model_type, ref_date, period_label)
    if df_pred is None:
        st.warning(f"참조일({ref_date})에 해당 피크 시간대 데이터가 없습니다. 다른 시간대를 선택해 주세요.")
        st.stop()
    info["ref_date"] = ref_date

    if info["excluded"]:
        st.caption(f"⚠️ 재고 정보가 없는 대여소 {info['excluded']}곳은 제외했습니다.")
    if info["unmatched"]:
        st.caption(f"⚠️ 대여소 이름 파일과 매칭되지 않은 대여소 {info['unmatched']}곳은 ID로 표시합니다.")
    return df_pred, info


def station_row(df, station_id):
    """사이드바에서 고른 station_id에 해당하는 행(Series). 없으면 None."""
    if station_id is None:
        return None
    match = df[df["대여소ID"] == station_id]
    if match.empty:
        st.caption("선택한 대여소는 이번 조건에서 예측 결과가 없어 전체 보기로 표시합니다.")
        return None
    return match.iloc[0]


def make_base_map(df, selected=None):
    """전체 보기는 서울 전역, 대여소를 고르면 해당 위치로 확대 + 빨간 원으로 표시."""
    if selected is None:
        m = folium.Map(location=[df["위도"].mean(), df["경도"].mean()], zoom_start=12)
    else:
        m = folium.Map(location=[selected["위도"], selected["경도"]], zoom_start=16)
        folium.CircleMarker(
            location=[selected["위도"], selected["경도"]],
            radius=22,
            color="#e63946",
            weight=3,
            fill=False,
        ).add_to(m)
    return m


def map_key(prefix, selected):
    """선택이 바뀔 때마다 지도를 새로 그려서 확대 위치가 확실히 적용되도록 하는 key."""
    station = "all" if selected is None else selected["대여소ID"]
    return f"{prefix}_{model_type}_{selected_date}_{period_label}_{station}"


# ==========================================
# 5. 메인 화면 분기
# ==========================================

# ------------------------------------------
# Mode 1: 실시간 지도 시각화
# ------------------------------------------
if mode == "1. 실시간 수급 불균형 지도":
    df_predicted, info = get_predictions()
    window_text = f"{selected_date} {period_label} ({info['hour_min']:02d}~{info['hour_max']:02d}시)"
    st.subheader(f"📊 [{model_type} 모델] {window_text} 기준 대여소별 Net-flow(반납-대여) 예측 현황")

    shortage_cnt = len(df_predicted[df_predicted["위험상태"].str.contains("부족")])
    overflow_cnt = len(df_predicted[df_predicted["위험상태"].str.contains("포화")])

    st.markdown("---")
    left_col, right_col = st.columns([2, 1])

    with left_col:
        title_col, stats_col = st.columns([1, 1.2], vertical_alignment="center")
        title_col.markdown(
            f'<div class="compact-heading">🗺️ 대여소별 위험도 지도 ({model_type})</div>',
            unsafe_allow_html=True,
        )
        # 값과 변화 태그를 한 줄에 놓기 위해 st.metric 대신 직접 만든 HTML을 사용 (줄바꿈 없이 한 줄로 조립)
        stats_html = (
            '<div class="stat-row">'
            '<div class="stat"><div class="stat-label">분석 대여소</div>'
            f'<div class="stat-line"><span class="stat-value">{len(df_predicted)}개소</span></div></div>'
            '<div class="stat"><div class="stat-label">부족 위험 '
            '<span class="stat-help" title="예측 잔여 자전거가 3대 이하라 대여가 어려운 대여소">?</span></div>'
            f'<div class="stat-line"><span class="stat-value">{shortage_cnt}곳</span>'
            '<span class="stat-pill down">↓ -경고</span></div></div>'
            '<div class="stat"><div class="stat-label">포화 위험 '
            '<span class="stat-help" title="예측 잔여 자전거가 거치대 여유 3칸 이하라 반납이 어려운 대여소">?</span></div>'
            f'<div class="stat-line"><span class="stat-value">{overflow_cnt}곳</span>'
            '<span class="stat-pill up">↑ +주의</span></div></div>'
            '</div>'
        )
        stats_col.markdown(stats_html, unsafe_allow_html=True)
        selected_station = station_row(df_predicted, selected_station_id)
        m = make_base_map(df_predicted, selected_station)

        for _, row in df_predicted.iterrows():
            is_selected = (
                selected_station is not None and row["대여소ID"] == selected_station["대여소ID"]
            )
            if "부족" in row["위험상태"]:
                color = "red"
                icon_type = "exclamation-sign"
            elif "포화" in row["위험상태"]:
                color = "blue"
                icon_type = "info-sign"
            else:
                color = "green"
                icon_type = "ok-sign"

            popup_text = (
                f"<div style='width:230px'>"
                f"<b>{row['대여소명']}</b><br>"
                f"• 지역: {row['클러스터']} ({row['구분']})<br>"
                f"• 상태: <b>{row['위험상태']}</b><br>"
                f"• 예측 대여 {row['예측_대여']:.1f}대 / 반납 {row['예측_반납']:.1f}대<br>"
                f"• 예측 순수요(Net-flow): {row['순수요_NetFlow']:+.1f}대<br>"
                f"• <b>예측 잔여: {row['예측_잔여수량']}대</b> / (총 {int(row['총거치대수'])}석)"
                f"</div>"
            )

            folium.Marker(
                location=[row["위도"], row["경도"]],
                popup=folium.Popup(popup_text, max_width=300, show=is_selected),
                icon=folium.Icon(color=color, icon=icon_type, prefix="glyphicon"),
            ).add_to(m)

        st_folium(
            m,
            width="100%",
            height=450,
            returned_objects=[],
            key=map_key("map1", selected_station),
        )

    with right_col:
        st.markdown(
            '<div class="compact-heading">📋 대여소별 Net-flow 상세표</div>', unsafe_allow_html=True
        )
        table_cols = ["대여소명", "순수요_NetFlow", "위험상태"]
        if info["has_actual"]:
            table_cols.insert(2, "실제_NetFlow")
        table = df_predicted[table_cols].copy()
        for c in ("순수요_NetFlow", "실제_NetFlow"):
            if c in table.columns:
                table[c] = table[c].round(1)
        table = table.rename(columns={"실제_NetFlow": "참조일_실제_NetFlow"})
        st.dataframe(one_based(table), use_container_width=True, height=400)
        if info["has_actual"]:
            st.caption(
                f"참조일_실제_NetFlow: 참조일({info['ref_date']}) 같은 시간대의 실제 (반납 - 대여) 합계로, "
                "모델이 그날을 얼마나 맞혔는지 비교하는 용도입니다."
            )

# ------------------------------------------
# Mode 2: K-Means 클러스터 분석
# ------------------------------------------
elif mode == "2. K-Means 클러스터 분석":
    df_predicted, info = get_predictions()
    st.subheader("🏙️ 군집 분석(K-Means) 기반 업무 밀집가 수급 불균형 클러스터")
    st.markdown("수급 불균형이 심각하게 연동되는 주요 거점 클러스터(예: 여의도, 마곡, 강남 등 업무 밀집가)를 그룹화하여 분석합니다.")

    cluster_group = (
        df_predicted.groupby("클러스터")
        .agg(
            대여소수=("대여소ID", "count"),
            평균순수요=("순수요_NetFlow", "mean"),
            평균잔여수량=("예측_잔여수량", "mean"),
        )
        .round(1)
        .reset_index()
    )

    st.dataframe(one_based(cluster_group), use_container_width=True)

    selected_station = station_row(df_predicted, selected_station_id)
    m_cluster = make_base_map(df_predicted, selected_station)
    for _, row in df_predicted.iterrows():
        is_selected = (
            selected_station is not None and row["대여소ID"] == selected_station["대여소ID"]
        )
        folium.CircleMarker(
            location=[row["위도"], row["경도"]],
            radius=10,
            popup=folium.Popup(
                f"<b>{row['클러스터']}</b><br>{row['대여소명']}", max_width=300, show=is_selected
            ),
            color="orange",
            fill=True,
            fill_color="orange",
            fill_opacity=0.7,
        ).add_to(m_cluster)

    st_folium(
        m_cluster,
        width="100%",
        height=400,
        returned_objects=[],
        key=map_key("map2", selected_station),
    )

# ------------------------------------------
# Mode 3: 시간대별 대여·반납 예측 (선 그래프)
# ------------------------------------------
elif mode == "3. 시간대별 대여·반납 예측 (선 그래프)":
    ref_date = get_reference_date()
    daily = run_daily_prediction(model_type, ref_date)
    if daily is None:
        st.warning(f"참조일({ref_date})의 데이터가 없습니다. 다른 날짜를 선택해 주세요.")
        st.stop()

    weekday = WEEKDAYS[selected_date.weekday()]
    st.subheader(f"📈 [{model_type} 모델] {selected_date}({weekday}) 시간대별 예측 대여·반납 대수")

    picked_id = selected_station_id
    if picked_id is not None and not (daily["대여소ID"] == picked_id).any():
        st.caption("선택한 대여소는 이 날짜 데이터가 없어 전체 대여소 합계를 보여드립니다.")
        picked_id = None
    hourly = hourly_totals(daily, picked_id)

    if picked_id is None:
        target_name = "전체 대여소 합계"
    else:
        target_name = daily.loc[daily["대여소ID"] == picked_id, "대여소명"].iloc[0]
    st.markdown(f"##### {target_name}")
    st.line_chart(
        hourly,
        color=["#e63946", "#1d7ae0"],  # 예측 대여 = 빨강, 예측 반납 = 파랑
        x_label="시간 (시)",
        y_label="자전거 대수",
        height=420,
    )
    with st.expander("시간별 예측값 표 보기"):
        st.dataframe(hourly.round(1), use_container_width=True)

# ------------------------------------------
# Mode 4: 따릉이 길찾기 (출발지 근처 대여 → 도착지 근처 반납)
# ------------------------------------------
elif mode == "4. 가까운 대여소 찾기 (경로 지도)":
    if not find_data_file(STATIONS_FILE).exists():
        st.error(f"{STATIONS_FILE} 파일을 찾을 수 없습니다. data/ 폴더에 넣어 주세요.")
        st.stop()

    stations = load_station_locations()
    st.subheader("🧭 출발지 근처에서 빌려 도착지 근처에 반납하기")
    st.caption(
        "출발지·도착지를 장소 이름이나 주소로 검색하면 각각 가장 가까운 대여소를 찾고, "
        "대여소 사이 **자전거 탑승 예상 시간**을 보여 줍니다. 대상 대여소는 분석용 60곳입니다."
    )
    if not kakao_key():
        st.error(
            "카카오 API 키(KAKAO_REST_KEY)가 없습니다. 앱을 실행하기 전에 환경변수 `KAKAO_REST_KEY`를 설정하거나, "
            '`.streamlit/secrets.toml`에 `KAKAO_REST_KEY = "발급받은 REST API 키"`를 넣고 앱을 다시 실행해 주세요.'
        )
        st.stop()

    # 버튼을 눌렀을 때만 검색 (입력할 때마다 API를 부르지 않도록)
    with st.form("route_form"):
        f1, f2 = st.columns(2)
        q_origin = f1.text_input("출발지", placeholder="예: 서울역, 서울 중구 세종대로 110")
        q_dest = f2.text_input("도착지", placeholder="예: 광화문")
        submitted = st.form_submit_button("🔍 대여소 찾기", type="primary")
    if submitted:
        st.session_state["route_query"] = (q_origin.strip(), q_dest.strip())
    q_origin, q_dest = st.session_state.get("route_query", ("", ""))
    if not q_origin or not q_dest:
        st.info("출발지와 도착지를 모두 입력하고 '대여소 찾기'를 눌러 주세요.")
        st.stop()

    def pick_place(label, query, col):
        """카카오 검색 → 결과가 여러 개면 고르게 한다. 실패하면 이유를 보여 주고 None."""
        try:
            results = kakao_search(query)
        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code in (401, 403):
                col.error(
                    f"카카오 API 키가 올바르지 않거나 사용 권한이 없습니다(오류 {code}). "
                    "secrets.toml의 키와 카카오 개발자 사이트의 앱 설정을 확인해 주세요."
                )
            else:
                col.warning(f"'{query}' 검색에 실패했습니다(오류 {code}). 잠시 후 다시 시도해 주세요.")
            return None
        except Exception:
            col.warning(f"'{query}' 검색 중 카카오 서버에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요.")
            return None
        if not results:
            col.warning(f"'{query}' 검색 결과가 없습니다. 다른 이름이나 주소로 검색해 주세요.")
            return None
        labels = [f"{r['name']} · {r['address']}" if r["address"] else r["name"] for r in results]
        idx = col.selectbox(
            f"{label} 검색 결과", range(len(results)), format_func=lambda i: labels[i],
            key=f"pick_{label}_{query}",
        )
        return results[idx]

    p1, p2 = st.columns(2)
    origin = pick_place("출발지", q_origin, p1)
    dest = pick_place("도착지", q_dest, p2)
    if origin is None or dest is None:
        st.stop()

    forecast, forecast_note = station_forecast()

    def with_status(cands):
        """후보 표에 예측 재고 상태를 붙인다(예측이 없으면 '정보 없음')."""
        cands = cands.copy()
        if forecast is None:
            cands["예측상태"] = "정보 없음"
            cands["예측잔여"] = np.nan
        else:
            cands["예측상태"] = cands["station_id"].map(forecast["위험상태"]).fillna("정보 없음")
            cands["예측잔여"] = cands["station_id"].map(forecast["예측_잔여수량"])
        return cands

    rent_c = with_status(nearest_to_point(stations, origin["lat"], origin["lon"]))
    ret_c = with_status(nearest_to_point(stations, dest["lat"], dest["lon"]))
    for label, place, cands in (("출발지", origin, rent_c), ("도착지", dest, ret_c)):
        if cands.empty:
            st.error(
                f"{label} '{place['name']}'에서 직선 {fmt_distance(MAX_STATION_DIST_M)} 안에 대상 대여소가 없습니다. "
                "서울 시내의 다른 장소로 검색해 주세요."
            )
    if rent_c.empty or ret_c.empty:
        st.stop()

    def choose(cands, bad_word, role):
        """가까운 순서로, 예측상 bad_word(부족/포화)가 아닌 첫 대여소를 고른다."""
        ok = cands[~cands["예측상태"].str.contains(bad_word)]
        if ok.empty:
            st.warning(f"{role} 후보 {len(cands)}곳이 모두 '{bad_word} 위험'으로 예측됩니다. 가장 가까운 곳을 표시합니다.")
            return cands.iloc[0]
        pick = ok.iloc[0]
        if pick["station_id"] != cands.iloc[0]["station_id"]:
            st.caption(
                f"💡 {role}: 가장 가까운 '{cands.iloc[0]['station_name']}'은(는) '{bad_word} 위험'으로 예측되어 "
                f"다음으로 가까운 '{pick['station_name']}'을(를) 골랐습니다."
            )
        return pick

    rent = choose(rent_c, "부족", "대여")
    ret = choose(ret_c, "포화", "반납")
    o = (origin["lat"], origin["lon"])
    d = (dest["lat"], dest["lon"])
    s1 = (float(rent["lat"]), float(rent["lon"]))
    s2 = (float(ret["lat"]), float(ret["lon"]))

    same_station = rent["station_id"] == ret["station_id"]
    route = None if same_station else bike_leg(s1, s2)

    st.markdown(
        f"**🚲 대여:** {rent['station_name']} ({rent['gu']}) → **반납:** {ret['station_name']} ({ret['gu']})"
    )
    if same_station:
        st.info("출발지와 도착지에서 가장 가까운 대여소가 같습니다. 자전거를 탈 필요 없이 걸어가는 편이 낫습니다.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("자전거 탑승 예상 시간", f"약 {max(1, round(route['duration_s'] / 60))}분",
                  help="대여소 → 대여소 구간만의 시간입니다. 걷는 시간과 대여·반납 절차 시간은 포함하지 않습니다.")
        m2.metric("자전거 이동거리", fmt_distance(route["distance_m"]))
        m3.metric("출발지 → 대여소 (직선)", fmt_distance(rent["straight_m"]))
        m4.metric("반납 대여소 → 도착지 (직선)", fmt_distance(ret["straight_m"]))
        if route["estimated"]:
            st.warning(
                "자전거 경로 서버에 연결하지 못해 직선거리 × 1.3, 평균 시속 15km로 추정했습니다(점선). "
                "잠시 후 다시 시도하면 실제 경로로 계산합니다."
            )
        if rent["straight_m"] + ret["straight_m"] > route["distance_m"]:
            st.info("대여소까지 걷는 거리가 자전거 이동거리보다 깁니다. 걸어가는 편이 나을 수 있습니다.")
    for role, row in (("대여", rent), ("반납", ret)):
        if row["straight_m"] > WARN_STATION_DIST_M:
            st.warning(f"{role} 대여소까지 직선으로 {fmt_distance(row['straight_m'])} 떨어져 있습니다. 걸어서 시간이 더 걸릴 수 있습니다.")
    st.caption(f"ℹ️ {forecast_note}")

    with st.expander("가까운 대여소 후보 보기 (직선거리 순)"):
        t1, t2 = st.columns(2)
        for col, title, cands in ((t1, "출발지 근처 (대여)", rent_c), (t2, "도착지 근처 (반납)", ret_c)):
            col.markdown(f"**{title}**")
            view = pd.DataFrame({
                "대여소": cands["station_name"],
                "구": cands["gu"],
                "직선거리": cands["straight_m"].map(fmt_distance),
                "예측 상태": cands["예측상태"],
                "예측 잔여(대)": cands["예측잔여"],
            })
            col.dataframe(one_based(view), width="stretch")

    # 지도: 자전거 경로(빨강), 걷는 구간은 직선 점선(회색, 참고용)
    path = [list(s1), list(s2)] if route is None else route["path"]
    pts = [list(o), list(d), list(s1), list(s2), *path]
    # fit_bounds는 화면 맞춤 애니메이션이 중간에 끊겨 엉뚱한 위치로 열리는 경우가 있어, 중심·확대 단계를 직접 계산한다
    center, zoom = fit_view(pts)
    m5 = folium.Map(location=center, zoom_start=zoom)
    for _, r in stations[~stations["station_id"].isin([rent["station_id"], ret["station_id"]])].iterrows():
        folium.CircleMarker(location=[r["lat"], r["lon"]], radius=4, color="#8d99ae", fill=True,
                            fill_opacity=0.8, tooltip=f"{r['station_name']} ({r['gu']})").add_to(m5)
    for a, b in ((o, s1), (s2, d)):
        folium.PolyLine([list(a), list(b)], color="#8d99ae", weight=3, dash_array="4 6",
                        tooltip="걷는 구간 (직선, 참고용)").add_to(m5)
    if route is not None:
        style = {"color": "#e63946", "weight": 5, "opacity": 0.9}
        if route["estimated"]:
            style["dash_array"] = "8"
        folium.PolyLine(path, tooltip=f"자전거 약 {fmt_distance(route['distance_m'])}", **style).add_to(m5)
    for loc, tip, color, icon in (
        (o, f"출발지: {origin['name']}", "green", "home"),
        (s1, f"대여: {rent['station_name']}", "red", "star"),
        (s2, f"반납: {ret['station_name']}", "blue", "flag"),
        (d, f"도착지: {dest['name']}", "black", "map-marker"),
    ):
        folium.Marker(location=list(loc), tooltip=tip,
                      icon=folium.Icon(color=color, icon=icon, prefix="glyphicon")).add_to(m5)
    st_folium(m5, width="100%", height=480, returned_objects=[], key="map5_route")
    st.caption(
        "주소 검색: Kakao 로컬 API · 경로: © OpenStreetMap contributors (ODbL) · OSRM, routing.openstreetmap.de (FOSSGIS). "
        "공개 데모 서버를 쓰므로 비상업·소규모 용도로만 사용하세요."
    )
