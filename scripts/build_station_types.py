"""대여소 유형(K-Means) 계산 — 메뉴 2 '대여소 유형 지도'에서 읽는 파일을 만든다.

[역할] 60개 대여소를 '평일에 자전거가 어떻게 들고 나는지'가 비슷한 것끼리 4개 유형으로 묶고,
       유형마다 누구나 알아들을 이름·설명을 붙여 data/station_types.csv, data/station_types_meta.json으로 저장한다.
       앱은 이 파일을 읽기만 한다(앱 실행 때 K-Means를 다시 돌리지 않으므로 화면이 빠르고 결과가 매번 같다).
[원리] 1) 2025년 평일(공휴일 제외) 시간별 대여·반납 기록(data/longterm_c/history_hourly.parquet)으로 대여소마다 4가지 값을 만든다.
          - am_ratio  : 출근 07~09시 (반납-대여)/(반납+대여). +면 자전거가 들어오고(회사 쪽), -면 빠져나간다(집 쪽).
          - pm_ratio  : 퇴근 18~20시 같은 계산.
          - log_volume: 하루 평균 이용(대여+반납) 건수의 로그. 붐비는 정도(로그로 큰 값의 영향을 줄인다).
          - peak_share: 하루 이용 중 출퇴근 6시간이 차지하는 비율. 출퇴근 전용인지, 하루 종일 쓰이는지.
       2) 단위가 다른 값을 같은 무게로 비교하도록 표준화(평균 0, 표준편차 1)한 뒤 K-Means(k=4, seed 42, 20번 시작)로 묶는다.
          k는 2~6의 실루엣 점수(묶음이 얼마나 뚜렷한지)를 함께 기록한다. 4는 2(최고점)와 거의 같은 점수이면서
          이용자에게 의미 있는 구분(집·회사·역 앞 등)을 줄 수 있어 고른 값이다.
       3) 유형 이름은 사람이 묶음마다 손으로 붙이지 않고, 각 묶음의 중심값으로 정한 규칙으로 붙인다(name_clusters).
주의: 유형 이름은 '이용 흐름'으로 붙인 별명이다. 실제 토지 용도(회사·주거)를 확인한 것은 아니다.
      2025년 기록을 쓰므로, 학습 2021~2023 기준으로 정한 station_group(역세권·업무지구·주거지)과는 별개의 분류다.
실행: .venv\\Scripts\\python.exe scripts\\build_station_types.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / 'data/longterm_c/history_hourly.parquet'
HOLI = ROOT / 'data/longterm_c/holidays.csv'
OUT_CSV = ROOT / 'data/station_types.csv'
OUT_META = ROOT / 'data/station_types_meta.json'
YEAR, K, SEED = 2025, 4, 42
FEATURES = ['am_ratio', 'pm_ratio', 'log_volume', 'peak_share']
AM_HOURS, PM_HOURS = (7, 9), (18, 20)

# [역할] 유형별 화면 표시 정보. [원리] 이름 규칙(name_clusters)이 정한 key로 찾는다. 색은 색약에도 구분되는 4색.
TYPE_INFO = {
    'office': dict(emoji='🏢', name='출근길 도착지 (회사 밀집)', color='#d62728',
                   desc='아침에 자전거가 많이 들어오고 저녁에 빠져나가요. 출근 시간엔 반납 자리가, 퇴근 시간엔 자전거가 모자라기 쉬워요.'),
    'mixed': dict(emoji='🏬', name='출근길 도착지 (회사·생활 혼합)', color='#ff7f0e',
                  desc='아침엔 자전거가 들어오는 편이지만, 낮에도 꾸준히 쓰여 회사 밀집 지역보다 덜 쏠려요.'),
    'busy': dict(emoji='🚇', name='하루 종일 붐비는 곳 (역 앞 등)', color='#9467bd',
                 desc='들고 나는 양이 비슷하지만 이용 자체가 가장 많아요. 자전거가 빨리 바뀌므로 도착 직전에 한 번 더 확인하세요.'),
    'home': dict(emoji='🏠', name='출근길 출발지 (주거 지역)', color='#1f77b4',
                 desc='아침에 자전거를 타고 나가고 저녁에 돌아와요. 출근 시간엔 자전거가, 퇴근 시간엔 반납 자리가 모자라기 쉬워요.'),
    'balanced': dict(emoji='⚖️', name='들고 나는 게 비슷한 곳', color='#2ca02c',
                     desc='출퇴근 쏠림이 크지 않아 비교적 안정적이에요.'),
}


def station_features(h, holidays):
    """[역할] 대여소별 특징 4개 계산. [원리] 2025년 평일·비공휴일만 남기고, 시간대별로 합계를 낸 뒤 비율로 바꾼다.
    합계로 비율을 내므로 하루하루 비율의 평균보다 이용이 적은 날의 흔들림에 덜 민감하다."""
    ts = h['timestamp'].dt.tz_localize(None)
    keep = (ts.dt.year == YEAR) & (ts.dt.day_of_week < 5) & ~ts.dt.strftime('%Y-%m-%d').isin(holidays)
    h = h.loc[keep].assign(hour=ts[keep].dt.hour, day=ts[keep].dt.date)

    def total(mask):
        return h.loc[mask].groupby('station_id')[['rentals', 'returns']].sum(min_count=1)

    am, pm, day = total(h.hour.between(*AM_HOURS)), total(h.hour.between(*PM_HOURS)), total(h.hour >= 0)
    n_days = h.dropna(subset=['rentals']).groupby('station_id')['day'].nunique()
    f = pd.DataFrame({
        'am_ratio': (am.returns - am.rentals) / (am.returns + am.rentals),
        'pm_ratio': (pm.returns - pm.rentals) / (pm.returns + pm.rentals),
        'daily_volume': (day.rentals + day.returns) / n_days,
        'peak_share': (am.sum(axis=1) + pm.sum(axis=1)) / day.sum(axis=1),
        # 화면용: 출근 시간 '반납 ÷ 대여' 배수(1보다 크면 들어오는 쪽). 비율보다 직관적인 표현.
        'am_return_per_rental': am.returns / am.rentals,
        'pm_return_per_rental': pm.returns / pm.rentals,
        'n_days': n_days,
    })
    f['log_volume'] = np.log(f['daily_volume'])
    return f


def name_clusters(centers):
    """[역할] 묶음 번호 → 유형 key. [원리] 묶음 중심값(원래 단위)을 보고 정해진 순서로 이름을 붙인다.
      ① 출근 비율이 가장 낮고 음수(-0.2 미만)인 묶음 = home(출근길 출발지)
      ② 남은 것 중 이용량이 가장 많고 출근 쏠림이 작은(|am|<0.3) 묶음 = busy(종일 붐빔)
      ③ 남은 것 중 출근 비율이 양수인 묶음: 가장 높은 것 = office, 그다음 = mixed
      ④ 그 밖 = balanced
    규칙이 데이터와 안 맞으면(예: ①이 없음) 해당 이름을 건너뛰므로, 같은 이름이 두 번 붙지 않는다."""
    left = list(centers.index)
    names = {}
    c = centers.loc[left].sort_values('am_ratio')
    if c.iloc[0]['am_ratio'] < -0.2:
        names[c.index[0]] = 'home'
        left.remove(c.index[0])
    c = centers.loc[left]
    calm = c[c['am_ratio'].abs() < 0.3]
    if len(calm):
        names[calm['daily_volume'].idxmax()] = 'busy'
        left.remove(calm['daily_volume'].idxmax())
    pos = centers.loc[left].query('am_ratio > 0').sort_values('am_ratio', ascending=False)
    for key, idx in zip(['office', 'mixed'], pos.index):
        names[idx] = key
        left.remove(idx)
    for idx in left:
        names[idx] = 'balanced'
    assert len(set(names.values())) == len(names), f'유형 이름 중복: {names}'
    return names


def main():
    h = pd.read_parquet(HIST)
    holidays = set(pd.read_csv(HOLI)['date'])
    f = station_features(h, holidays).dropna(subset=FEATURES)
    X = StandardScaler().fit_transform(f[FEATURES])
    sil = {k: round(float(silhouette_score(X, KMeans(k, n_init=20, random_state=SEED).fit_predict(X))), 3)
           for k in range(2, 7)}
    km = KMeans(K, n_init=20, random_state=SEED).fit(X)
    f['cluster'] = km.labels_
    centers = f.groupby('cluster')[FEATURES + ['daily_volume', 'am_return_per_rental', 'pm_return_per_rental']].mean()
    names = name_clusters(centers)
    f['type_key'] = f['cluster'].map(names)
    for col in ('emoji', 'name', 'color'):
        f[f'type_{col}'] = f['type_key'].map(lambda k: TYPE_INFO[k][col])

    out = f.reset_index()[['station_id', 'type_key', 'type_emoji', 'type_name', 'type_color',
                           'am_ratio', 'pm_ratio', 'am_return_per_rental', 'pm_return_per_rental',
                           'daily_volume', 'peak_share', 'n_days']].round(3)
    out.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

    summary = []
    for cl, key in names.items():
        m = f[f.cluster == cl]
        # 유형 대표값은 중앙값: 대여소 몇 곳의 극단값에 끌려가지 않게 한다.
        summary.append({**TYPE_INFO[key], 'key': key, 'n_stations': int(len(m)),
                        'am_return_per_rental': round(float(m['am_return_per_rental'].median()), 2),
                        'pm_return_per_rental': round(float(m['pm_return_per_rental'].median()), 2),
                        'daily_volume': round(float(m['daily_volume'].median()), 0),
                        'peak_share': round(float(m['peak_share'].median()), 2)})
    order = ['home', 'office', 'mixed', 'busy', 'balanced']
    summary.sort(key=lambda s: order.index(s['key']))
    meta = {'method': 'K-Means (표준화한 특징 4개, k=4, seed 42, n_init 20)',
            'period': f'{YEAR}년 평일(공휴일 제외)', 'features': FEATURES,
            'feature_ko': {'am_ratio': '출근 07~09시 (반납-대여)/(반납+대여)', 'pm_ratio': '퇴근 18~20시 같은 계산',
                           'log_volume': '하루 평균 이용 건수(로그)', 'peak_share': '하루 이용 중 출퇴근 6시간 비율'},
            'silhouette_by_k': sil, 'k': K,
            # [수정 2026-11-17] 화면에 그대로 나가는 문장이라 내부 용어(station_group)를 쓰지 않는다.
            'note': '유형 이름은 이용 흐름으로 붙인 별명이며, 실제 토지 용도(회사·주거)를 확인한 것은 아니에요. '
                    '예측 모델에 쓴 대여소 구분(역 주변·회사·주거, 2021~2023년 기록 기준)과는 별개의 분류예요.',
            'types': summary}
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print('실루엣(k별):', sil)
    for s in summary:
        print(f"{s['emoji']} {s['name']}: {s['n_stations']}곳, 출근 반납/대여 {s['am_return_per_rental']}배, "
              f"퇴근 {s['pm_return_per_rental']}배, 하루 {s['daily_volume']:.0f}건, 출퇴근 비중 {s['peak_share']}")


if __name__ == '__main__':
    main()
