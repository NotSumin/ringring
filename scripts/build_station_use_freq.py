"""'내 대여소 카드'의 시간대별 대여·반납 판정에 쓰는 작년 빈도표(data/station_use_freq.parquet)를 만든다.

[역할] 앱이 2026년 날짜 d의 출퇴근 시각 h에 대해 "작년 이 시기에 이 대여소가 얼마나 자주 비었나 / 매우 붐볐나"를
       바로 찾아볼 수 있게, 2025년(작년) 기준 빈도를 미리 계산해 둔다. 앱은 읽기만 한다.
[원리] 검증 스크립트(experiments/longterm_c/eval_use_forecast.py)와 같은 정의를 쓴다.
  - 비었음  = h시와 h+1시 재고 중 작은 값 ≤ 3대
  - 매우 붐빔 = h시와 h+1시 재고 중 큰 값 ≥ 거치대 수 × 1.5
  - 날 종류: 평일(work) / 주말·공휴일(off)
  - 빈도: 기준일(= 예측일 − 364일)의 ±28일 안, 같은 날 종류·같은 시각 날들 중 사건이 일어난 날의 비율. n_ref = 그 날 수.
  검증(2023→2024, 2024→2025): 대여 🟢(빈도 <0.3)일 때 실제로 빈 비율 11%, 🔴(≥0.6) 62~63%;
  반납(1.5배) 🟢 15%, 🔴 70~73%. 그날 예측을 섞으면 오히려 덜 맞아 섞지 않는다.
실행: .venv\\Scripts\\python.exe scripts\\build_station_use_freq.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / 'data/longterm_c/history_hourly.parquet'
STATIC = ROOT / 'data/longterm_c/station_static.csv'
HOLI = ROOT / 'data/longterm_c/holidays.csv'
OUT = ROOT / 'data/station_use_freq.parquet'
HOURS = [7, 8, 9, 18, 19, 20]
WINDOW_DAYS = 28
EMPTY_MAX = 3          # 이 대수 이하면 '비었음'
CROWD_RATIO = 1.5      # 거치대 수의 이 배수 이상이면 '매우 붐빔'


def main():
    h = pd.read_parquet(HIST, columns=['station_id', 'timestamp', 'inventory']).dropna(subset=['inventory'])
    h['station_id'] = h['station_id'].astype(str)
    ts = h['timestamp'].dt.tz_localize(None)
    h = h.assign(date=pd.to_datetime(ts.dt.date), hour=ts.dt.hour)
    dock = pd.read_csv(STATIC, dtype={'station_id': str}).set_index('station_id')['dock_count']
    hol = set(pd.read_csv(HOLI)['date'])

    # 날짜 × 시각 재고표 → 시각 h마다 (h, h+1) 최소·최대
    w = h.pivot_table(index=['station_id', 'date'], columns='hour', values='inventory', aggfunc='first')
    parts = []
    for hr in HOURS:
        pair = pd.concat([w.get(hr), w.get(hr + 1)], axis=1)
        parts.append(pd.DataFrame({'hour': hr, 'lo': pair.min(axis=1, skipna=False), 'hi': pair.max(axis=1, skipna=False)}))
    e = pd.concat(parts).reset_index().dropna(subset=['lo', 'hi'])
    e['empty'] = (e['lo'] <= EMPTY_MAX).astype(float)
    e['crowd'] = (e['hi'] >= e['station_id'].map(dock) * CROWD_RATIO).astype(float)
    e['kind'] = np.where((e['date'].dt.day_of_week >= 5) | e['date'].dt.strftime('%Y-%m-%d').isin(hol), 'off', 'work')

    # 2025년 모든 날짜(기준일)에 대해 ±28일 이동 평균. 같은 날 종류끼리만 모은다.
    days = pd.date_range('2025-01-01', '2025-12-31', freq='D')
    out = []
    for (sid, hr, k), g in e.groupby(['station_id', 'hour', 'kind']):
        g = g.set_index('date').sort_index()[['empty', 'crowd']]
        roll = g.rolling(f'{2 * WINDOW_DAYS + 1}D', center=True, min_periods=1)
        r = roll.mean().assign(n_ref=roll['empty'].count()).reindex(days, method='nearest')
        r = r.assign(station_id=sid, hour=hr, kind=k)
        out.append(r.rename_axis('ref_date').reset_index())
    f = pd.concat(out, ignore_index=True).rename(columns={'empty': 'f_empty', 'crowd': 'f_crowd'})
    f['n_ref'] = f['n_ref'].astype(int)
    f[['station_id', 'hour', 'kind', 'ref_date', 'f_empty', 'f_crowd', 'n_ref']].to_parquet(OUT, index=False)
    print(f'저장: {OUT} | {len(f):,}행 | 대여소 {f.station_id.nunique()}곳 | n_ref 5 미만 {int((f.n_ref < 5).sum())}행')


if __name__ == '__main__':
    main()
