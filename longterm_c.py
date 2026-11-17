"""C안 날짜 예측 모듈 — 최근 기록 없이 '1년 전 기록 + 유사일 + 날씨 + 달력'으로 하루치 대여·반납을 예측한다.

[역할] 앱에서 2026년 날짜를 고르면, 그날 60개 대여소 × 24시간의 예측 대여·반납 수를 돌려준다.
[원리] 학습 때(머신러닝 팀프로젝트/experiments/longterm_c)와 똑같은 규칙으로 입력을 만든다.
       - 이용·재고 기록은 예측 시각보다 364일(52주) 이상 지난 값만 쓴다 → 2025년 기록으로 2026년 예측 가능.
       - 유사일: 작년 같은 시기(D-392~D-364일) 중 휴일 여부가 같고 비 여부·최고기온이 비슷한 날 최대 5일의 평균.
       - 날씨: 기상청 예보 연결 전에는 사용자가 입력한 '날씨 가정'(기본값은 평년값)을 쓴다.
       - 모델은 "작년과 같은 수준"으로 학습됐으므로, 2026 하반기에는 상반기 통계 증감률의 절반(α 0.5)만큼 낮춰 준다.
필요 파일: models/longterm_c/{model_rentals.txt, model_returns.txt, preprocess.json, meta.json}
          data/longterm_c/{history_hourly.parquet, daily_weather.csv, station_static.csv, holidays.csv}
"""
from dataclasses import dataclass
from pathlib import Path

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

TZ = 'Asia/Seoul'
WEEK_H = 168
YEAR_H = 52 * WEEK_H            # 8,736시간 = 364일 (요일이 같음)
PRECIP_LABELS = ['없음', '약한 비 (반나절 5mm 이하)', '보통 비 (5~20mm)', '강한 비 (20mm 초과)']


# [역할] 앱이 쓰는 파일을 한 번에 읽어 두는 상자. [원리] 앱은 st.cache_resource로 한 번만 만든다.
@dataclass
class Bundle:
    models: dict
    inputs: list
    categories: dict
    meta: dict
    hist: pd.DataFrame          # (station_id, timestamp) 색인, 열: rentals, returns, inventory (coverage 밖은 결측)
    daily: pd.DataFrame         # 날짜 색인, 열: day_of_week, is_offday, temp_min_day, temp_max_day, rain_am, rain_pm ...
    static: pd.DataFrame        # 대여소 고정 정보
    holidays: pd.DataFrame      # date(문자열), name, holiday_type


def load_bundle(model_dir, data_dir):
    """[역할] 묶음 파일 읽기. [원리] LightGBM은 한글 경로 문제를 피하려고 파일 내용을 문자열로 읽어 모델을 만든다."""
    model_dir, data_dir = Path(model_dir), Path(data_dir)
    pre = json.loads((model_dir / 'preprocess.json').read_text(encoding='utf-8'))
    models = {t: lgb.Booster(model_str=(model_dir / f'model_{t}.txt').read_text(encoding='utf-8'))
              for t in ('rentals', 'returns')}
    hist = pd.read_parquet(data_dir / 'history_hourly.parquet')
    hist['station_id'] = hist.station_id.astype(str)
    # 시각 정밀도를 나노초로 맞춘다: parquet는 마이크로초로 저장될 수 있어, 앱이 만든 시각과 색인 비교가 어긋나지 않게 한다
    hist['timestamp'] = hist.timestamp.astype('datetime64[ns, Asia/Seoul]')
    hist = hist.set_index(['station_id', 'timestamp']).sort_index()
    daily = pd.read_csv(data_dir / 'daily_weather.csv', encoding='utf-8-sig', parse_dates=['date'])
    for c in ('rain_am', 'rain_pm'):
        daily[c] = daily[c].map({True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0})
    daily = daily.set_index('date').sort_index()
    static = pd.read_csv(data_dir / 'station_static.csv', encoding='utf-8-sig', dtype={'station_id': str})
    holidays = pd.read_csv(data_dir / 'holidays.csv', encoding='utf-8-sig', dtype=str)
    meta = json.loads((model_dir / 'meta.json').read_text(encoding='utf-8'))
    return Bundle(models, pre['inputs'], pre['categories'], meta, hist, daily, static, holidays)


def calendar_for_day(day, holidays):
    """[역할] 하루의 달력 입력(공휴일·연휴 길이·연휴 전후일·근로자의 날) 계산.
    [원리] 학습 자료(build_weather_calendar.py)와 같은 정의: 주말 또는 공휴일 = 쉬는 날,
           쉬는 날이 이어지는 길이 = 연휴 길이, 쉬는 날 바로 전/후 평일 표시. 앞뒤 20일을 함께 계산해 연휴를 판정한다."""
    day = pd.Timestamp(day).normalize()
    days = pd.DataFrame({'date': pd.date_range(day - pd.Timedelta(days=20), day + pd.Timedelta(days=20), freq='D')})
    h = holidays.assign(date=pd.to_datetime(holidays.date))
    days = days.merge(h, on='date', how='left')
    days['is_holiday'] = days['name'].notna()
    days['is_weekend'] = days.date.dt.day_of_week >= 5
    days['is_offday'] = days.is_holiday | days.is_weekend
    days['is_day_before_offday'] = ~days.is_offday & days.is_offday.shift(-1).astype(bool)
    days['is_day_after_offday'] = ~days.is_offday & days.is_offday.shift(1).astype(bool)
    block = (days.is_offday != days.is_offday.shift()).cumsum()
    days['offday_block_len'] = days.groupby(block).is_offday.transform('size').where(days.is_offday, 0)
    days['is_labor_day'] = (days.date.dt.month == 5) & (days.date.dt.day == 1)
    return days.set_index('date').loc[day]


def climatology(daily, day, window=7):
    """[역할] 날씨 가정의 기본값(평년값). [원리] 2021~2025년 중 같은 날짜 ±7일의 최저·최고기온 평균."""
    d = daily.reset_index()
    doy = d.date.dt.dayofyear
    target = pd.Timestamp(day).dayofyear
    diff = np.minimum((doy - target).abs(), 366 - (doy - target).abs())
    near = d[diff <= window]
    return round(float(near.temp_min_day.mean()), 1), round(float(near.temp_max_day.mean()), 1)


def similar_dates(daily, day, is_offday, temp_max, rain_am, rain_pm, k=5, window=28, min_n=3):
    """[역할] 유사일 고르기. [원리] D-392~D-364일 중 휴일 여부가 같은 날에 점수를 매겨 낮은 순 최대 5일.
    점수 = 오전·오후 비 여부가 다르면 1건당 10점 + 최고기온 차이(℃) + 요일이 다르면 0.5점. 후보가 3일 미만이면 사용 안 함."""
    day = pd.Timestamp(day).normalize()
    lo, hi = day - pd.Timedelta(days=364 + window), day - pd.Timedelta(days=364)
    c = daily.loc[lo:hi]
    c = c[c.is_offday.astype(bool).eq(bool(is_offday))].dropna(subset=['temp_max_day', 'rain_am', 'rain_pm'])
    if len(c) < min_n:
        return []
    score = ((c.rain_am != rain_am).astype(int) + (c.rain_pm != rain_pm).astype(int)) * 10 \
        + (c.temp_max_day - temp_max).abs() + (c.day_of_week != day.dayofweek) * 0.5
    return list(score.nsmallest(k).index)


def weather_inputs(w):
    """[역할] 날씨 가정(최저·최고기온, 오전·오후 강수 구간 0~3, 눈 여부)을 모델 입력 열로 바꾼다.
    [원리] 강수 구간이 1 이상이면 '비 옴'(rain_am/pm=1). 최저기온 0℃ 이하이면서 비가 오면 눈 추정(snow_est_day=1) — 학습 때 정의.
    [추가 2026-11-16 눈] w['snow']=1(기상청 눈 예보 또는 사용자가 '눈' 선택)이면 기온과 관계없이 눈으로 본다.
           눈인데 강수 구간이 둘 다 0이면 '약한 강수'(구간 1)로 올린다 — 눈도 강수(물 환산)이므로.
           주의: 학습 자료의 눈 추정일은 모두 최저기온 0℃ 이하였으므로, 영상 기온의 눈은 모델이 처음 보는 조합이다."""
    am, pm = int(w['precip_am_cat']), int(w['precip_pm_cat'])
    snow = bool(w.get('snow', 0))
    if snow and am == 0 and pm == 0:
        am = pm = 1
    rain = am > 0 or pm > 0
    return {'temp_min_day': float(w['temp_min']), 'temp_max_day': float(w['temp_max']),
            'rain_am': float(am > 0), 'rain_pm': float(pm > 0), 'precip_am_cat': float(am), 'precip_pm_cat': float(pm),
            'snow_est_day': float(rain and (w['temp_min'] <= 0 or snow)), 'snow_est_prev_day': float(w.get('snow_prev', 0))}


def is_snow_day(w):
    """[역할] 화면 안내용: 그날이나 전날이 눈(추정 포함)인지. [원리] weather_inputs와 같은 규칙으로 판단한다."""
    x = weather_inputs(w)
    return bool(x['snow_est_day'] or x['snow_est_prev_day'])


def build_features(day, weather, b, static=None):
    """[역할] 하루치(대여소 × 24시간) 모델 입력 표를 만든다.
    [원리] ① 달력 ② 대여소 정보 ③ 364일 전 기준 1년 전 기록 ④ 유사일 평균 ⑤ 날씨 가정을 붙인다.
           static을 따로 주면(검증용) 그 값을, 아니면 묶음의 최신 대여소 정보를 쓴다."""
    day = pd.Timestamp(day).normalize()
    static = b.static if static is None else static
    ts = pd.date_range(day, periods=24, freq='h', tz=TZ)
    f = static.merge(pd.DataFrame({'timestamp': ts}), how='cross')
    f['hour'] = f.timestamp.dt.hour
    f['day_of_week'] = f.timestamp.dt.dayofweek
    f['month'] = f.timestamp.dt.month
    f['is_weekend'] = f.day_of_week >= 5
    f['commute_period'] = np.select([f.hour.between(7, 9), f.hour.between(18, 20)], ['morning', 'evening'], 'other')
    cal = calendar_for_day(day, b.holidays)
    for c in ('is_holiday', 'is_offday', 'is_day_before_offday', 'is_day_after_offday', 'offday_block_len', 'is_labor_day'):
        f[c] = cal[c]
    f['holiday_type'] = cal['holiday_type'] if isinstance(cal['holiday_type'], str) else np.nan

    # ③ 1년 전 기록: 52·53·54·55주 전 같은 요일·시각. 2개 이상 있을 때만 평균.
    sid = f.station_id.astype(str)
    lags = {}
    for k in range(4):
        idx = pd.MultiIndex.from_arrays([sid, f.timestamp - pd.Timedelta(hours=YEAR_H + WEEK_H * k)])
        lags[k] = b.hist.reindex(idx)
    for col in ('rentals', 'returns', 'inventory'):
        m = pd.concat({k: lags[k][col].reset_index(drop=True) for k in lags}, axis=1)
        if col != 'inventory':
            f[f'{col}_lag_52w'] = m[0].to_numpy()
        f[f'{col}_ly_4w_mean'] = m.mean(axis=1).where(m.notna().sum(axis=1) >= 2).to_numpy()

    # ⑤ 날씨 가정
    wx = weather_inputs(weather)
    for c, v in wx.items():
        f[c] = v

    # ④ 유사일: 고른 날짜들의 같은 대여소·같은 시각 평균(기록 있는 값만), simday_n = 대여 값이 있는 날 수
    sims = similar_dates(b.daily, day, cal['is_offday'], wx['temp_max_day'], wx['rain_am'], wx['rain_pm'])
    if sims:
        parts = []
        for s in sims:
            idx = pd.MultiIndex.from_arrays([sid, f.timestamp - (day - pd.Timestamp(s)).to_pytimedelta()])
            parts.append(b.hist.reindex(idx)[['rentals', 'returns']].reset_index(drop=True))
        stack = pd.concat(parts, keys=range(len(parts)))
        f['rentals_simday_mean'] = stack.rentals.groupby(level=1).mean().to_numpy()
        f['returns_simday_mean'] = stack.returns.groupby(level=1).mean().to_numpy()
        f['simday_n'] = stack.rentals.groupby(level=1).count().astype(float).to_numpy()
    else:
        # 학습 때도 유사일 후보가 3일 미만이면 세 열 모두 결측(NaN)이었다 → 똑같이 결측으로 둔다
        f['rentals_simday_mean'] = f['returns_simday_mean'] = f['simday_n'] = np.nan
    f.attrs['similar_dates'] = [pd.Timestamp(s).date() for s in sims]
    return f


def to_model_input(f, b):
    """[역할] 입력 표를 LightGBM이 받는 형태로 변환. [원리] 범주형은 학습 때 범주 목록으로 고정(처음 보는 값은 결측), 나머지는 실수."""
    X = pd.DataFrame(index=f.index)
    for c in b.inputs:
        if c in b.categories:
            X[c] = pd.Categorical(f[c].astype(str).where(f[c].notna()), categories=b.categories[c])
        else:
            X[c] = f[c].astype('float64')
    return X


def h2_factor(day, b, target):
    """[역할] 2026 하반기 보정 배율. [원리] 상반기 통계 증감률^0.5. 2026년 7~12월에만 적용, 그 밖은 1."""
    day = pd.Timestamp(day)
    if day.year == 2026 and day.month >= 7:
        return float(b.meta['h2_factor_2026'][target])
    return 1.0


# ==========================================
# 추가 모델(XGBoost·GRU) — 2026-11-16
# ==========================================
# [역할] 날짜 예측에서 LightGBM 말고 XGBoost·GRU도 고를 수 있게 한다.
# [원리] 세 모델 모두 같은 입력 표(build_features)를 쓰고, 모델마다 학습 때와 똑같은 전처리만 다르게 적용한다.
#        학습 코드: 머신러닝 팀프로젝트/experiments/longterm_c/train_extra_models.py (전처리 함수는 그 파일과 같아야 한다).
EXTRA_FILES = {'XGBoost': ('xgb_{t}.json', 'xgb_{t}_preprocess.json'),
               'GRU': ('gru_{t}.keras', 'gru_{t}_preprocess.json')}


def available_models(model_dir):
    """[역할] 화면의 모델 선택지. [원리] LightGBM은 기본, 나머지는 대여·반납 파일이 모두 있을 때만 보인다."""
    model_dir = Path(model_dir)
    names = ['LightGBM']
    for name, (m, p) in EXTRA_FILES.items():
        if all((model_dir / x.format(t=t)).exists() for t in ('rentals', 'returns') for x in (m, p)):
            names.append(name)
    return names


def load_extra(model_dir, name):
    """[역할] XGBoost·GRU 모델과 전처리 정보 읽기. [원리] 무거운 라이브러리(tensorflow)는 그 모델을 고를 때만 불러온다.
    XGBoost는 한글 경로 문제를 피하려고 파일 내용을 메모리로 읽어 불러온다."""
    model_dir = Path(model_dir)
    out = {}
    for t in ('rentals', 'returns'):
        m_name, p_name = (x.format(t=t) for x in EXTRA_FILES[name])
        pre = json.loads((model_dir / p_name).read_text(encoding='utf-8'))
        if name == 'XGBoost':
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model(bytearray((model_dir / m_name).read_bytes()))
        else:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_dir / m_name, compile=False)
        out[t] = (model, pre)
    return out


def xgb_input(f, pre):
    """[역할] XGBoost 입력 표. [원리] 학습과 같게: 범주형은 학습 범주 목록으로 고정(처음 보는 값은 결측), 나머지는 실수."""
    X = pd.DataFrame(index=f.index)
    for c in pre['inputs']:
        X[c] = pd.Categorical(f[c].astype(str).where(f[c].notna()), categories=pre['categories'][c]) \
            if c in pre['categories'] else f[c].astype('float64')
    return X


def gru_input(f, pre):
    """[역할] GRU 입력 배열(대여소별 하루 24시간 순서). [원리] 학습과 같게: 숫자는 학습 평균·표준편차로 맞추고 결측은 0,
    결측 여부 표시 열을 붙이며, 범주는 학습 때 번호(처음 보는 값 0). f는 대여소·시각 순으로 하루 24행씩이어야 한다."""
    assert len(f) % 24 == 0 and (f.timestamp.dt.hour.to_numpy().reshape(-1, 24) == np.arange(24)).all()
    X = f[pre['nums']].astype('float64')
    Z = (X - pd.Series(pre['mean'])) / pd.Series(pre['std'])
    num = np.concatenate([Z.fillna(0).to_numpy(), X[pre['miss_cols']].isna().to_numpy(dtype='float64')], axis=1)
    cat = np.stack([f[c].astype(str).where(f[c].notna()).map(pre['cat_maps'][c]).fillna(0).to_numpy(dtype='int32')
                    for c in pre['cats']], axis=1)
    n = len(f) // 24
    return num.reshape(n, 24, -1).astype('float32'), cat.reshape(n, 24, -1)


def model_predict(f, b, model_name, extra, target):
    """[역할] 고른 모델로 입력 표 f의 예측값(음수는 0). [원리] 모델 종류에 맞는 전처리 후 예측."""
    if model_name == 'LightGBM':
        return np.clip(b.models[target].predict(to_model_input(f, b)), 0, None)
    model, pre = extra[target]
    if model_name == 'XGBoost':
        import xgboost as xgb
        d = xgb.DMatrix(xgb_input(f, pre), enable_categorical=True)
        return np.clip(model.predict(d, iteration_range=(0, int(pre['best_iteration']))), 0, None)
    num, cat = gru_input(f, pre)
    return np.clip(model.predict((num, cat), batch_size=1024, verbose=0).reshape(-1), 0, None)


def predict_day(day, weather, b, model_name='LightGBM', extra=None):
    """[역할] 하루치 예측. [원리] 입력 표 → 고른 모델 예측(음수는 0) → 하반기 보정.
    돌려주는 표: station_id, timestamp, hour, commute_period, pred_rentals, pred_returns, inventory_ly(작년 같은 시각 평균 재고).
    [추가 2026-11-16] model_name: 'LightGBM'(기본)·'XGBoost'·'GRU'. 추가 모델은 load_extra 결과를 extra로 넘긴다."""
    f0 = build_features(day, weather, b)
    sims = f0.attrs['similar_dates']          # 정렬 전에 꺼내 둔다(정렬하면 attrs가 사라질 수 있음)
    f = f0.sort_values(['station_id', 'timestamp']).reset_index(drop=True)   # GRU는 대여소별 24시간 순서가 필요
    out = f[['station_id', 'timestamp', 'hour', 'commute_period']].copy()
    for t in ('rentals', 'returns'):
        out[f'pred_{t}'] = model_predict(f, b, model_name, extra, t) * h2_factor(day, b, t)
    out['inventory_ly'] = f['inventory_ly_4w_mean']
    out.attrs['similar_dates'] = sims
    return out
