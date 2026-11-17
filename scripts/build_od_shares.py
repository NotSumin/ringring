"""운영자 보기 '이동 흐름' 화면용 이동 비율표(data/od_top_partners.csv)를 만든다.

[역할] 60개 대여소마다 출퇴근 시간에 "빌린 자전거가 어디로 가는지(나가는 흐름)", "반납된 자전거가 어디서 오는지(들어오는 흐름)"의
       상위 10곳과 비율을 미리 계산해 둔다. 앱은 이 비율 × 그날 예측 대여(반납) 수로 화살표 굵기를 정한다.
[원리] 이동 기록(od_flow.parquet: 출발·도착 대여소, 시각, 건수)에서 기준 연도의 평일/주말·공휴일 × 출근(7~9시)/퇴근(18~20시)별로
       대여소 쌍의 건수를 더한 뒤, 같은 출발(또는 도착) 대여소의 합계로 나눠 비율을 만든다. 상위 10곳만 남기고 나머지는 '그 밖'으로 본다.
       상대 대여소 이름·좌표는 대여소 정보(station_master.parquet)의 가장 최근 값을 쓴다(60곳 밖 대여소 포함).
       60곳 화면 이름은 앱과 같게 data/station_names_2026.csv의 display_name을 우선 쓴다.
실행: .venv\\Scripts\\python.exe scripts\\build_od_shares.py "<od_flow.parquet>" "<station_master.parquet>" "<공휴일 csv>" [기준연도]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/od_top_partners.csv'
TOP_N = 10
PERIODS = {'morning': (7, 9), 'evening': (18, 20)}


def shares(od, ids, year, holidays, top_n=TOP_N):
    """[역할] 기준 연도의 대여소별 상위 상대 대여소와 비율. 앱과 검증 스크립트가 같이 쓴다."""
    ts = od['timestamp'].dt.tz_localize(None)
    od = od[ts.dt.year == year].assign(ts=ts[ts.dt.year == year])
    hr = od['ts'].dt.hour
    od = od.assign(period=np.select([hr.between(*PERIODS['morning']), hr.between(*PERIODS['evening'])],
                                    ['morning', 'evening'], ''),
                   kind=np.where((od['ts'].dt.day_of_week >= 5) | od['ts'].dt.strftime('%Y-%m-%d').isin(holidays), 'off', 'work'),
                   day=od['ts'].dt.date)
    od = od[od['period'] != '']
    n_days = od.groupby('kind')['day'].nunique()
    rows = []
    for direction, me, other in (('out', 'origin', 'destination'), ('in', 'destination', 'origin')):
        g = od[od[me].isin(ids)].groupby([me, 'period', 'kind', other])['trips'].sum().rename('trips').reset_index()
        g['total'] = g.groupby([me, 'period', 'kind'])['trips'].transform('sum')
        g['share'] = g['trips'] / g['total']
        g = g.sort_values('trips', ascending=False).groupby([me, 'period', 'kind']).head(top_n)
        g['rank'] = g.groupby([me, 'period', 'kind']).cumcount() + 1
        g['per_day'] = g['trips'] / g['kind'].map(n_days)
        rows.append(g.rename(columns={me: 'station_id', other: 'partner_id'}).assign(direction=direction))
    return pd.concat(rows, ignore_index=True)


def main(od_path, master_path, holiday_path, year=2025):
    ids = set(pd.read_csv(ROOT / 'data/selected_stations_60.csv', dtype={'station_id': str})['station_id'].str.lstrip('0'))
    od = pd.read_parquet(od_path)
    od['origin'] = od['origin'].astype(str).str.lstrip('0')
    od['destination'] = od['destination'].astype(str).str.lstrip('0')
    hol = set(pd.read_csv(holiday_path)['date'])
    t = shares(od, ids, int(year), hol)
    m = pd.read_parquet(master_path, columns=['station_id', 'station_name', 'latitude', 'longitude', 'snapshot_at'])
    m['station_id'] = m['station_id'].astype(str).str.lstrip('0')
    m = m.sort_values('snapshot_at').groupby('station_id').last()
    names = pd.read_csv(ROOT / 'data/station_names_2026.csv', encoding='utf-8-sig', dtype={'station_id': str}).set_index('station_id')['display_name']
    t['partner_name'] = t['partner_id'].map(names).fillna(t['partner_id'].map(m['station_name'])).fillna(t['partner_id']).astype(str).str.strip()  # 원본 이름 앞뒤 공백 제거
    t['partner_lat'] = t['partner_id'].map(m['latitude'])
    t['partner_lon'] = t['partner_id'].map(m['longitude'])
    t['partner_in_60'] = t['partner_id'].isin(ids)
    t['base_year'] = int(year)
    cols = ['station_id', 'direction', 'period', 'kind', 'rank', 'partner_id', 'partner_name', 'partner_lat', 'partner_lon',
            'partner_in_60', 'share', 'per_day', 'trips', 'total', 'base_year']
    t[cols].round({'share': 4, 'per_day': 2}).to_csv(OUT, index=False, encoding='utf-8-sig')
    print(f'저장: {OUT} | {len(t):,}행 | 좌표 없는 상대 {int(t.partner_lat.isna().sum())}행')


if __name__ == '__main__':
    main(*sys.argv[1:])
