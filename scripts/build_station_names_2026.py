"""대여소 화면 이름을 2026년 공식 이름으로 맞추는 표(data/station_names_2026.csv)를 만든다.

[역할] 앱의 대여소 목록(data/selected_stations_60.csv)은 2021년 기준 이름이라, 그 뒤 개명된 곳은 이용자가 모르는
       옛 이름으로 보인다(예: 1012 '서울 상운차량' → '천호역효성해링턴타워 앞'). 원래 목록은 그대로 두고,
       화면에 보여 줄 이름만 따로 표로 만든다(앱이 읽어 이름만 바꿔 끼운다. 좌표·예측은 그대로).
[원리] 서울시 '공공자전거 대여소별 이용정보(월별)' 2026년 1~6월 파일의 '대여소명'("번호. 이름")에서 번호와 이름을 떼어
       가장 최근 달(6월)의 이름을 쓴다. 이름이 바뀌었지만 좌표로 같은 곳인지 확인하지 못한 대여소(LOCATION_UNVERIFIED)는
       혼동을 막으려고 '새 이름 (옛 옛이름)' 형태로 둔다.
실행: .venv\\Scripts\\python.exe scripts\\build_station_names_2026.py "<월별 이용정보 26.1-6 CSV 경로>"
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STATIONS = ROOT / 'data/selected_stations_60.csv'
OUT = ROOT / 'data/station_names_2026.csv'
# 2025-06까지는 좌표·이름이 같았는데 2026 파일에서만 이름이 바뀌어, 위치를 옮겼는지 확인하지 못한 대여소
LOCATION_UNVERIFIED = {'374'}
# 2026 파일에서만 '뒤 → 앞'처럼 한 단어가 바뀐 곳: 이름은 새 이름으로 쓰되, 기록용으로 '미확인' 메모만 남긴다
NOTE_ONLY_UNVERIFIED = {'834'}


def main(stat_csv):
    ours = pd.read_csv(STATIONS, encoding='utf-8-sig', dtype={'station_id': str})
    ours['station_id'] = ours['station_id'].str.lstrip('0')
    try:
        d = pd.read_csv(stat_csv, encoding='utf-8-sig')
    except UnicodeDecodeError:
        d = pd.read_csv(stat_csv, encoding='cp949')
    parts = d['대여소명'].astype(str).str.extract(r'^\s*0*(\d+)\.\s*(.*)$')
    d = d.assign(station_id=parts[0], name=parts[1].str.strip())
    latest = (d[d['station_id'].isin(ours['station_id'])]
              .sort_values('기준년월').groupby('station_id').last())
    out = ours[['station_id', 'station_name']].rename(columns={'station_name': 'name_2021'})
    out['name_2026'] = out['station_id'].map(latest['name'])
    out['month'] = out['station_id'].map(latest['기준년월'])
    changed = out['name_2026'].notna() & (out['name_2026'] != out['name_2021'].str.strip())
    out['display_name'] = out['name_2021'].str.strip()
    out.loc[changed, 'display_name'] = out.loc[changed, 'name_2026']
    unv = changed & out['station_id'].isin(LOCATION_UNVERIFIED)
    out.loc[unv, 'display_name'] = out.loc[unv, 'name_2026'] + ' (옛 ' + out.loc[unv, 'name_2021'].str.strip() + ')'
    out['note'] = ''
    out.loc[changed, 'note'] = '2026 공식 이름으로 변경'
    out.loc[unv, 'note'] = '이름 변경, 위치 이전 여부 미확인(최신 대여소 정보로 확인 필요)'
    minor = changed & out['station_id'].isin(NOTE_ONLY_UNVERIFIED)
    out.loc[minor, 'note'] = '2026 이름으로 변경(뒤→앞), 위치 미세 이동 여부 미확인'
    out.to_csv(OUT, index=False, encoding='utf-8-sig')
    print(f'저장: {OUT} | 이름 바뀐 곳 {int(changed.sum())}곳')
    print(out[changed][['station_id', 'name_2021', 'display_name']].to_string(index=False))


if __name__ == '__main__':
    main(sys.argv[1])
