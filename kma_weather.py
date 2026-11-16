"""기상청 예보 → C안 모델의 '날씨 입력'으로 바꾸는 모듈.

[역할] 고른 날짜의 서울 날씨 예보(최저·최고기온, 오전·오후 강수)를 기상청 API에서 받아,
       C안 모델이 학습한 형태(temp_min, temp_max, precip_am_cat, precip_pm_cat, snow_prev)로 돌려준다.
[원리] - 단기예보(getVilageFcst): 오늘~약 5일 뒤. 시간별 강수량(PCP)을 오전(0~11시)·오후(12~23시)로 더해
         학습 때와 같은 강수 구간(0 없음, 1 ≤5mm, 2 ≤20mm, 3 >20mm)으로 바꾼다. 최저·최고기온은 TMN/TMX, 없으면 시간별 기온(TMP)의 최저·최고.
       - 중기예보(getMidLandFcst·getMidTa): 단기예보가 없는 날. 강수량이 없고 강수확률·날씨 문구만 있으므로
         강수확률 60% 이상이거나 문구에 비·눈·소나기가 있으면 '약한 비'(구간 1)로 본다(가정 — 팀 결정 사항).
       - 모델은 '실제로 온 날씨'로 학습됐다. 예보가 틀리면 예측도 함께 틀린다.
       - 키(공공데이터포털 서비스 키)는 환경변수 KMA_SERVICE_KEY 또는 Streamlit secrets에서 읽는다(코드에 적지 않는다).
주의: 중기예보 서비스는 2026-11-12 개편 공지가 있어 응답 항목 이름이 바뀌었을 수 있다. 그래서 항목 이름을
      정규식으로 찾아 읽고, 못 읽으면 None을 돌려 앱이 평년값으로 대신하게 한다.
"""
import os
import re
from datetime import date, datetime, timedelta
from urllib.parse import unquote

import requests

SHORT_URL = 'https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst'
MID_LAND_URL = 'https://apis.data.go.kr/1360000/MidFcstInfoService/getMidLandFcst'
MID_TA_URL = 'https://apis.data.go.kr/1360000/MidFcstInfoService/getMidTa'
SEOUL_NX, SEOUL_NY = 60, 127          # 서울 종로구 격자(학습에 쓴 ASOS 108 서울 지점과 같은 지역)
MID_LAND_REG, MID_TA_REG = '11B00000', '11B10101'   # 중기 육상: 서울·인천·경기 / 중기 기온: 서울
RAIN_PROB_THRESHOLD = 60              # 중기예보 강수확률(%) 이 값 이상이면 비로 본다(가정)
SHORT_BASE_TIMES = [2, 5, 8, 11, 14, 17, 20, 23]   # 단기예보 발표 시각(하루 8번), 발표 약 10분 뒤 제공
TIMEOUT_S = 8


def service_key():
    """[역할] 서비스 키 찾기. [원리] 환경변수 → Streamlit secrets 순서. 포털의 '인코딩 키'를 넣어도 되도록 한 번 디코딩한다."""
    key = os.environ.get('KMA_SERVICE_KEY', '').strip()
    if not key:
        try:
            import streamlit as st
            key = str(st.secrets['KMA_SERVICE_KEY']).strip()
        except Exception:
            key = ''
    return unquote(key) if key else None


def _get_items(url, params):
    """[역할] 기상청 API 호출 후 item 목록만 꺼낸다.
    [원리] 정상 응답은 JSON(response.header.resultCode == '00'). 키 오류는 HTTP 403 + OpenAPI_ServiceResponse로 온다.
    [보안] requests의 오류 문구에는 요청 주소(서비스 키 포함)가 들어가므로 그대로 내보내지 않는다.
           화면에 보여 줄 오류는 서비스 이름과 기상청이 준 사유만 담은 새 문구로 만든다."""
    name = url.rsplit('/', 1)[-1]
    try:
        resp = requests.get(url, params={**params, 'dataType': 'JSON'}, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        raise RuntimeError(f'{name} 연결 실패({type(e).__name__})') from None
    try:
        data = resp.json()
    except ValueError:
        raise RuntimeError(f'{name} 응답 형식 오류(HTTP {resp.status_code})') from None
    err = data.get('OpenAPI_ServiceResponse', {}).get('cmmMsgHeader')
    if err:   # 포털 단계 오류: 키 미등록·활용신청 안 함·호출 한도 초과 등
        raise RuntimeError(f"{name} 키 오류: {err.get('returnAuthMsg') or err.get('errMsg')}")
    header = data.get('response', {}).get('header', {})
    if header.get('resultCode') != '00':
        raise RuntimeError(f"{name} 오류 {header.get('resultCode')}: {header.get('resultMsg')}")
    items = data['response']['body']['items']
    return items.get('item', []) if isinstance(items, dict) else []


def short_base(now):
    """[역할] 지금 받을 수 있는 가장 최근 단기예보 발표 시각. [원리] 발표 10분 뒤부터 제공되므로 10분을 빼고 고른다."""
    t = now - timedelta(minutes=10)
    for h in reversed(SHORT_BASE_TIMES):
        if t.hour >= h:
            return t.date(), h
    return t.date() - timedelta(days=1), 23


def parse_pcp(value):
    """[역할] 단기예보 강수량(PCP) 글자를 숫자(mm)로. [원리] '강수없음'=0, '1mm 미만'=0.5, 'a~bmm'=가운데 값, 'Nmm 이상'=N.
    5일째 정성값('약한 비'·'비'·'강한 비')은 시간당 기준(3mm 미만 / 3~15mm / 15mm 이상)의 대표값 1·8·15mm로 본다(가정)."""
    v = str(value).strip()
    if v in ('', '강수없음', '0', '0.0', '-'):
        return 0.0
    if '미만' in v:
        return 0.5
    qual = {'약한 비': 1.0, '비': 8.0, '강한 비': 15.0}
    if v in qual:
        return qual[v]
    nums = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', v)]
    if not nums:
        return 0.0
    return sum(nums) / len(nums)


def precip_cat(mm):
    """[역할] 반나절 강수 합계(mm) → 학습 때 강수 구간. [원리] 0 없음, (0,5] 약, (5,20] 보통, 20 초과 강."""
    if mm <= 0:
        return 0
    if mm <= 5:
        return 1
    if mm <= 20:
        return 2
    return 3


def parse_sno(value):
    """[역할] 단기예보 신적설(SNO) 글자를 cm로. [원리] '적설없음'·'0'=0, '1cm 미만'=0.5, 'a~bcm'=가운데 값, 'Ncm 이상'=N.
    5일째 정성값 '보통 눈'(시간당 1cm 미만)=0.5, '많은 눈'(1cm 이상)=2로 본다(가정)."""
    v = str(value).strip()
    if v in ('', '적설없음', '0', '0.0', '-'):
        return 0.0
    if '미만' in v or v in ('눈', '보통 눈'):
        return 0.5
    if v == '많은 눈':
        return 2.0
    nums = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', v)]
    return sum(nums) / len(nums) if nums else 0.0


SNOW_MM_PER_CM = 1.0   # 신적설 1cm ≈ 강수(물) 1mm로 환산(눈:물 = 10:1이라는 일반적인 기준, 가정)


def short_daily(items):
    """[역할] 단기예보 item 목록 → 날짜별 날씨 요약. [원리] 시각 0~11시는 오전, 12~23시는 오후로 강수량을 더한다.
    [추가 2026-11-16 눈] 같은 시각의 강수형태(PTY)·신적설(SNO)도 함께 본다.
      - PTY 2(비/눈)·3(눈) 또는 신적설 > 0 이면 그 시각은 '눈'으로 표시한다.
      - 눈일 때 강수량(PCP)이 비어 있을 수 있으므로, 그 시각 강수량 = max(PCP, 신적설cm × 1mm, 0.5mm).
        (눈이 올 때 PCP가 어떻게 오는지 실제 응답으로 확인하지 못해, 눈을 놓치지 않는 쪽으로 계산한다)"""
    hours = {}
    days = {}
    for it in items:
        d = datetime.strptime(it['fcstDate'], '%Y%m%d').date()
        h = int(it['fcstTime'][:2])
        rec = days.setdefault(d, {'tmn': None, 'tmx': None})
        cat, val = it['category'], it['fcstValue']
        if cat == 'TMN':
            rec['tmn'] = float(val)
        elif cat == 'TMX':
            rec['tmx'] = float(val)
        elif cat in ('PCP', 'PTY', 'SNO'):
            hours.setdefault((d, h), {})[cat] = val
    for d in days:
        days[d].update({'am': 0.0, 'pm': 0.0, 'pcp_n': 0, 'snow': 0})
    for (d, h), v in hours.items():
        if 'PCP' not in v:
            continue
        mm = parse_pcp(v['PCP'])
        sno = parse_sno(v.get('SNO', '0'))
        is_snow = str(v.get('PTY', '0')).strip() in ('2', '3') or sno > 0
        if is_snow:
            mm = max(mm, sno * SNOW_MM_PER_CM, 0.5)
            days[d]['snow'] = 1
        days[d]['am' if h < 12 else 'pm'] += mm
        days[d]['pcp_n'] += 1
    out = {}
    for d, r in days.items():
        # [수정 2026-11-16 실제 응답 확인] 오후 발표분에는 '오늘'이 15시 이후만 있고 TMN/TMX가 없다.
        # 하루 전체를 대표하는 최저·최고기온(TMN/TMX)이 모두 있는 날만 쓰고, 오늘은 그날 02시 발표분으로 채운다(fetch_forecast).
        if r['tmn'] is None or r['tmx'] is None or r['pcp_n'] < 4:
            continue
        # [수정] 3일째부터는 3시간 간격(하루 8개)으로 온다. 한 값이 3시간을 대표한다고 보고 3배로 더한다(가정).
        scale = 3.0 if r['pcp_n'] <= 8 else 1.0
        am, pm = r['am'] * scale, r['pm'] * scale
        out[d] = {'temp_min': round(r['tmn'], 1), 'temp_max': round(r['tmx'], 1),
                  'precip_am_cat': precip_cat(am), 'precip_pm_cat': precip_cat(pm),
                  'am_mm': round(am, 1), 'pm_mm': round(pm, 1), 'snow': r['snow']}
    return out


def mid_tmfc(now):
    """[역할] 지금 받을 수 있는 가장 최근 중기예보 발표 시각(06시·18시). [원리] 발표 30분 뒤부터 쓴다고 보고 고른다."""
    t = now - timedelta(minutes=30)
    if t.hour >= 18:
        return datetime(t.year, t.month, t.day, 18)
    if t.hour >= 6:
        return datetime(t.year, t.month, t.day, 6)
    y = t - timedelta(days=1)
    return datetime(y.year, y.month, y.day, 18)


def mid_daily(land, ta, tmfc):
    """[역할] 중기예보 → 날짜별 날씨 요약. [원리] 항목 이름 끝의 숫자 n = 발표일로부터 n일 뒤.
    taMin{n}/taMax{n}로 기온, rnSt{n}Am/Pm(또는 rnSt{n})·wf{n}Am/Pm(또는 wf{n})으로 비 여부를 정한다."""
    out = {}
    base = tmfc.date()
    for k, v in ta.items():
        m = re.fullmatch(r'taMin(\d+)', k)
        if not m:
            continue
        n = int(m.group(1))
        tmax = ta.get(f'taMax{n}')
        if v in (None, '') or tmax in (None, ''):
            continue

        def rainy(part):
            prob = land.get(f'rnSt{n}{part}', land.get(f'rnSt{n}'))
            text = str(land.get(f'wf{n}{part}', land.get(f'wf{n}', '')))
            try:
                p = float(prob)
            except (TypeError, ValueError):
                p = 0.0
            return int(p >= RAIN_PROB_THRESHOLD or any(w in text for w in ('비', '눈', '소나기')))

        # [추가 2026-11-16 눈] 날씨 문구(오전·오후 또는 하루)에 '눈'이 있으면 눈으로 표시('비/눈', '눈/비' 포함)
        snow = int(any('눈' in str(land.get(k, '')) for k in (f'wf{n}Am', f'wf{n}Pm', f'wf{n}')))
        out[base + timedelta(days=n)] = {'temp_min': float(v), 'temp_max': float(tmax),
                                         'precip_am_cat': rainy('Am'), 'precip_pm_cat': rainy('Pm'),
                                         'am_mm': None, 'pm_mm': None, 'snow': snow}
    return out


def fetch_forecast(key, now=None):
    """[역할] 단기·중기예보를 받아 날짜별 날씨 표(dict)와 출처 설명을 돌려준다.
    [원리] 단기예보를 우선 쓰고, 단기예보에 없는 날짜만 중기예보로 채운다. 한쪽이 실패해도 다른 쪽은 쓴다."""
    now = now or datetime.now()
    result, notes = {}, []
    # [수정] 가장 최근 발표분 + (그게 오늘 02시가 아니면) 오늘 02시 발표분도 받는다.
    #        최근 발표분을 우선 쓰고, 거기서 빠진 '오늘'처럼 하루가 다 안 담긴 날만 02시 발표분으로 채운다.
    bd, bh = short_base(now)
    bases = [(bd, bh)] + ([(now.date(), 2)] if (bd, bh) != (now.date(), 2) and bd == now.date() else [])
    for b_date, b_hour in bases:
        try:
            items = _get_items(SHORT_URL, {'serviceKey': key, 'pageNo': 1, 'numOfRows': 2000,
                                           'base_date': b_date.strftime('%Y%m%d'), 'base_time': f'{b_hour:02d}00',
                                           'nx': SEOUL_NX, 'ny': SEOUL_NY})
            for d, w in short_daily(items).items():
                result.setdefault(d, {**w, 'source': f'단기예보 {b_date:%m/%d} {b_hour:02d}시 발표'})
        except Exception as e:  # noqa: BLE001 — 실패 이유(키 없이 만든 문구)를 화면에 보여 주기 위해 모아 둔다
            notes.append(f'단기예보 실패: {e}')
    try:
        tm = mid_tmfc(now)
        p = {'serviceKey': key, 'pageNo': 1, 'numOfRows': 10, 'tmFc': tm.strftime('%Y%m%d%H00')}
        land = (_get_items(MID_LAND_URL, {**p, 'regId': MID_LAND_REG}) or [{}])[0]
        ta = (_get_items(MID_TA_URL, {**p, 'regId': MID_TA_REG}) or [{}])[0]
        for d, w in mid_daily(land, ta, tm).items():
            result.setdefault(d, {**w, 'source': f'중기예보 {tm:%m/%d} {tm:%H}시 발표(강수는 확률·문구로 판단)'})
    except Exception as e:  # noqa: BLE001
        notes.append(f'중기예보 실패: {e}')
    return result, notes


def to_model_weather(day, table):
    """[역할] 예보 표에서 그날 값을 C안 입력 형태로.
    [원리] snow = 기상청이 그날 눈(강수형태·신적설·문구)을 예보했는지. 전날이 '눈'이었거나 '0℃ 이하 + 강수'였으면 snow_prev=1."""
    w = table.get(day)
    if w is None:
        return None
    prev = table.get(day - timedelta(days=1))
    prev_rain = bool(prev) and (prev['precip_am_cat'] > 0 or prev['precip_pm_cat'] > 0)
    snow_prev = float(prev_rain and (prev['temp_min'] <= 0 or bool(prev.get('snow'))))
    return {'temp_min': w['temp_min'], 'temp_max': w['temp_max'], 'precip_am_cat': int(w['precip_am_cat']),
            'precip_pm_cat': int(w['precip_pm_cat']), 'snow_prev': snow_prev, 'snow': int(w.get('snow', 0)),
            'source': w['source']}
