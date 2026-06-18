# 📈 나스닥 백테스팅 (Nasdaq Backtesting)

나스닥 지수·ETF·미국 주식의 과거 데이터로 매매 전략을 **백테스트**하고,
결과를 차트로 확인하며, 만든 사이트를 **링크 하나로 공유**할 수 있는 웹앱입니다.

> ⚠️ 교육·연구용 도구입니다. 실제 투자 권유가 아니며, 과거 성과가 미래 수익을 보장하지 않습니다.

## ✨ 기능

- **역대 전체 기간** 백테스트 (예: 나스닥 종합지수는 1971년부터)
- **주봉 기준**(일봉·월봉도 선택 가능) — 지표는 봉 기준에 맞게 연율화
- 🔍 **전략 자동 탐색(Optimizer)**: 수십 개의 전략·파라미터 조합을 모두 백테스트하여
  **'매수 후 보유'를 이기는 전략**을 찾아 순위로 제시
- 📊 단일 전략 백테스트: **이동평균 교차(SMA)**, **RSI(시그널선 교차)**, **추세추종(MA)**, **매수 후 보유**
- 주요 지표: 총수익률, 연환산수익률(CAGR), 변동성, 샤프지수, 최대낙폭(MDD)
- 가격 차트의 매수/매도 시점 + 전략 vs 매수후보유 자산곡선(로그스케일) 비교, 결과 CSV 다운로드

## 🖥 내 PC에서 실행하기

```bash
# 1) 가상환경 생성 (선택)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2) 패키지 설치
pip install -r requirements.txt

# 3) 실행
streamlit run app.py
```

브라우저에서 자동으로 `http://localhost:8501` 이 열립니다.

## 🌐 인터넷에 공개 배포하기 (무료)

1. 이 저장소를 GitHub에 push 합니다.
2. <https://share.streamlit.io> 접속 → GitHub 로그인
3. **New app** → 저장소 `chanhee2330-prog/nasdaq` 선택 → 메인 파일 `app.py` → **Deploy**
4. 몇 분 뒤 `https://<앱이름>.streamlit.app` 주소가 생성됩니다. 이 링크를 공유하면 누구나 접속할 수 있습니다.

## 📂 구조

```
app.py              # Streamlit 앱 (UI + 백테스트 엔진)
requirements.txt    # 의존성
.streamlit/config.toml
README.md
```
