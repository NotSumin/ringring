# 백업과 되돌리기 (2026-11-12)

4번 메뉴를 "출발지·도착지 주소 검색 → 가까운 대여소 → 자전거 탑승 시간" 방식으로 바꾸기 전의 파일입니다.

| 백업 파일 | 원래 파일 | 비고 |
|---|---|---|
| `app_before_route_20261112.py` | `app.py` | 변경 전 원본 그대로 (git 커밋 `dcd84f7`과 같음) |
| `gitignore_before_route_20261112.txt` | `.gitignore` | `.streamlit/secrets.toml` 줄 추가 전 |

## 되돌리는 방법 (둘 중 하나)

1. 백업 파일 복사 (PowerShell, 이 저장소 폴더에서)

   ```powershell
   Copy-Item backup\app_before_route_20261112.py app.py -Force
   Copy-Item backup\gitignore_before_route_20261112.txt .gitignore -Force
   ```

2. git으로 되돌리기 (아직 커밋하지 않았을 때)

   ```powershell
   git restore app.py .gitignore
   ```

   이미 커밋했다면 `git restore --source dcd84f7 app.py`로 원본을 꺼낼 수 있습니다.

`.gitignore`는 되돌리지 않는 편이 안전합니다(되돌리면 카카오 키 파일이 Git에 올라갈 수 있음).
