"""
나스닥/반도체 백테스팅 웹앱
==========================
- 캔들차트 위에서 여러 매매 전략을 동시에 겹쳐(오버레이) 매수/매도 시점을 보고
- 전략별 수익률(자산곡선)을 한눈에 비교한다.
- 신호 종목(NQ=F, SOXX 등)으로 신호를 만들고, 실제 손익은 거래 종목(SOXL 등)으로 계산.
- 고급 검증: 전략 자동 탐색 + 워크포워드(과최적화 배제) 검증.

실행:  streamlit run app.py
"""

import warnings

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# 일부 네트워크(회사망·백신·방화벽의 SSL 검사)에서는 야후 인증서 체인을
# 공개 CA로 검증할 수 없어 데이터 다운로드가 실패한다. 그런 환경에서도
# 동작하도록 인증서 검증을 끈 curl_cffi 세션을 yfinance에 넘긴다.
warnings.filterwarnings("ignore", message="Unverified HTTPS request")


@st.cache_resource(show_spinner=False)
def _yf_session():
    try:
        from curl_cffi import requests as _creq
        return _creq.Session(impersonate="chrome", verify=False)
    except Exception:
        return None

from engine import (
    build_position, compute_metrics, map_position, optimize, run_backtest,
    signal_buy_and_hold, spec_label, synth_leverage_df, walk_forward,
)

st.set_page_config(page_title="나스닥/SOXL 백테스팅", page_icon="📈", layout="wide")

# 신호용 종목(추세 판단)
SIGNAL_TICKERS = {
    "나스닥 종합지수 (^IXIC)": "^IXIC",
    "나스닥100 (QQQ)": "QQQ",
    "나스닥 선물 (NQ=F)": "NQ=F",
    "반도체 지수 ETF (SOXX)": "SOXX",
    "직접 입력": "__custom__",
}
# 실제 거래 종목(레버리지 등)
TRADE_TICKERS = {
    "신호 종목과 동일 (1배)": "__same__",
    "합성 레버리지 (신호 지수 ×N, 전체기간)": "__synth__",
    "SOXL (반도체 3배 ↑)": "SOXL",
    "TQQQ (나스닥 3배 ↑)": "TQQQ",
    "QQQ (1배)": "QQQ",
    "SOXX (1배)": "SOXX",
    "직접 입력": "__custom__",
}
# 봉 기준 → (interval, 연간 봉 수, 데이터 기간) — 백테스트·자동탐색·워크포워드용
INTERVALS = {
    "시간봉 (약 2년)": ("1h", 1700, "730d"),
    "일봉 (전체기간)": ("1d", 252, "max"),
    "주봉 (전체기간)": ("1wk", 52, "max"),
    "월봉 (전체기간)": ("1mo", 12, "max"),
}
# 전략 연구실 타임프레임 → (interval, 연간 봉 수, 기간, 인트라데이 여부)
# 야후 제한: 분봉(1m) 최근 약 7일, 시간봉(1h) 약 2년. 일/주/월봉은 역대 전체.
LAB_TF = {
    "분봉 (1m·약7일)": ("1m", 252 * 390, "7d", True),
    "시간봉 (1h·약2년)": ("1h", 1700, "730d", True),
    "일봉 (전체기간)": ("1d", 252, "max", False),
    "주봉 (전체기간)": ("1wk", 52, "max", False),
    "월봉 (전체기간)": ("1mo", 12, "max", False),
}

# 전략 슬롯에서 고를 수 있는 전략 종류 + 슬롯별 색
LAB_STRATS = ["추세추종(MA)", "돌파+트레일", "SMA교차", "RSI"]
SLOT_COLORS = ["#2ca02c", "#ff7f0e", "#9467bd", "#17becf", "#bcbd22", "#e377c2"]
UP, DOWN = "#d62728", "#1f77b4"  # 한국식: 상승=빨강 / 하락=파랑


# ----------------------------------------------------------------------------
# 데이터
# ----------------------------------------------------------------------------
@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False, session=_yf_session())
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df


# ----------------------------------------------------------------------------
# 전략 슬롯 UI (사이드바) — 내 알고리즘을 슬롯마다 세밀하게 조정
# ----------------------------------------------------------------------------
def strategy_slot(i: int, default_type: str, default_on: bool) -> dict:
    color = SLOT_COLORS[i % len(SLOT_COLORS)]
    with st.sidebar.expander(f"전략 {i + 1}", expanded=default_on):
        st.markdown(f"<span style='color:{color};font-size:20px'>■</span> 차트 색",
                    unsafe_allow_html=True)
        on = st.checkbox("이 전략 겹쳐보기", value=default_on, key=f"on_{i}")
        stype = st.selectbox("전략 종류", LAB_STRATS,
                             index=LAB_STRATS.index(default_type), key=f"ty_{i}")
        sp = {"type": stype}
        valid, msg = True, ""
        if stype == "추세추종(MA)":
            sp["window"] = st.slider("이동평균 기간(봉)", 3, 300, 200, key=f"w_{i}",
                                     help="종가가 이 MA '위'면 보유, '아래'면 현금.")
            sp["buffer"] = st.slider("버퍼 밴드(%)", 0.0, 10.0, 5.0, 0.5, key=f"bf_{i}",
                                     help="MA를 이 % 이상 확실히 넘을 때만 매매(휩쏘↓). 3~5% 권장.") / 100.0
        elif stype == "돌파+트레일":
            sp["donchian_n"] = st.slider("돈치안 돌파기간(진입)", 5, 100, 20, key=f"dn_{i}")
            sp["atr_n"] = st.slider("ATR 기간", 5, 30, 14, key=f"an_{i}")
            sp["atr_k"] = st.slider("ATR 트레일 배수(클수록 길게 보유)", 1.0, 6.0, 3.0, 0.5, key=f"ak_{i}")
            sp["use_adx"] = st.checkbox("ADX 추세강도 필터", value=False, key=f"ua_{i}")
            if sp["use_adx"]:
                sp["adx_min"] = st.slider("최소 ADX", 10, 40, 20, key=f"am_{i}")
            sp["use_disp"] = st.checkbox("이격도 과열 필터", value=False, key=f"ud_{i}")
            if sp["use_disp"]:
                sp["disp_cap"] = st.slider("이격도 상한(이 위면 진입 금지)", 101, 140, 115, key=f"dc_{i}")
                sp["disp_n"] = st.slider("이격도 MA 기간", 5, 60, 20, key=f"dpn_{i}")
        elif stype == "SMA교차":
            sp["short"] = st.slider("단기 MA", 3, 100, 20, key=f"sh_{i}")
            sp["long"] = st.slider("장기 MA", 10, 300, 100, key=f"lo_{i}")
            if sp["short"] >= sp["long"]:
                valid, msg = False, "단기 MA는 장기 MA보다 작아야 합니다."
        elif stype == "RSI":
            sp["period"] = st.slider("RSI 기간", 5, 30, 14, key=f"rp_{i}")
            sp["ma_period"] = st.slider("RSI 시그널 MA", 2, 30, 9, key=f"rm_{i}")
            sp["oversold"] = st.slider("과매도", 10, 45, 30, key=f"ro_{i}")
            sp["overbought"] = st.slider("과매수", 55, 90, 70, key=f"rb_{i}")
            if sp["oversold"] >= sp["overbought"]:
                valid, msg = False, "과매도는 과매수보다 작아야 합니다."
        if on and not valid:
            st.warning(msg)
    return {"on": on and valid, "spec": sp, "color": color}


# ----------------------------------------------------------------------------
# 전략 연구실 차트: 캔들 + MA + 전략별 매수/매도 마커 + 전략별 자산곡선 비교
# ----------------------------------------------------------------------------
def build_lab_fig(trade_df, ticker, ma_periods, log_scale, intraday,
                  strat_results, bh_equity) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.6, 0.4],
                        subplot_titles=(f"{ticker} 캔들 + 전략별 매수(▲)/매도(▼) 시점",
                                        "전략별 자산 곡선 비교"))

    fig.add_trace(go.Candlestick(
        x=trade_df.index, open=trade_df["Open"], high=trade_df["High"],
        low=trade_df["Low"], close=trade_df["Close"], name=ticker,
        increasing_line_color=UP, decreasing_line_color=DOWN,
        increasing_fillcolor=UP, decreasing_fillcolor=DOWN, showlegend=False,
    ), row=1, col=1)

    for p in ma_periods:
        if len(trade_df) > p:
            fig.add_trace(go.Scatter(x=trade_df.index, y=trade_df["Close"].rolling(p).mean(),
                                     name=f"MA{p}", line=dict(width=1), opacity=0.65,
                                     legendgroup="ma"), row=1, col=1)

    for k, sr in enumerate(strat_results):
        res, color, label = sr["result"], sr["color"], sr["label"]
        pos = res["Position"]
        entries = res.index[pos.diff() > 0]
        exits = res.index[pos.diff() < 0]
        off = 0.018 * (k + 1)  # 전략마다 마커를 살짝 어긋나게 (겹침 방지)
        fig.add_trace(go.Scatter(
            x=entries, y=trade_df.loc[entries, "Low"] * (1 - off), mode="markers",
            name=f"{label} · 매수/매도", legendgroup=label,
            marker=dict(symbol="triangle-up", color=color, size=10,
                        line=dict(width=0.6, color="white"))), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=exits, y=trade_df.loc[exits, "High"] * (1 + off), mode="markers",
            legendgroup=label, showlegend=False,
            marker=dict(symbol="triangle-down", color=color, size=10,
                        line=dict(width=0.6, color="white"))), row=1, col=1)
        fig.add_trace(go.Scatter(x=res.index, y=res["StrategyEquity"], name=label,
                                 legendgroup=label, line=dict(color=color, width=1.7)),
                      row=2, col=1)

    fig.add_trace(go.Scatter(x=bh_equity.index, y=bh_equity, name="매수 후 보유",
                             line=dict(color="#999999", dash="dash", width=1.4)), row=2, col=1)

    fig.update_layout(height=840, hovermode="x unified", legend=dict(orientation="h"),
                      margin=dict(t=46, b=10, l=10, r=10), xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="가격", row=1, col=1)
    fig.update_yaxes(title_text="자산(시작=1)", type="log", row=2, col=1)
    if log_scale:
        fig.update_yaxes(type="log", row=1, col=1)
    fig.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.05, row=2, col=1)
    if intraday:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    else:
        fig.update_xaxes(rangeselector=dict(buttons=[
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(count=5, label="5Y", step="year", stepmode="backward"),
            dict(step="all", label="전체"),
        ]), row=1, col=1)
    return fig


def ticker_picker(box, tickers, label, default_idx, custom_default):
    """공통 종목 선택 위젯 → (표시 티커 문자열, 원시 코드)."""
    lab = box.selectbox(label, list(tickers.keys()), index=default_idx)
    code = tickers[lab]
    if code == "__custom__":
        code = box.text_input(f"{label} 직접 입력", value=custom_default).strip().upper()
    return code


# ============================================================================
# 사이드바
# ============================================================================
st.sidebar.header("⚙️ 설정")
view = st.sidebar.radio(
    "화면", ["📈 전략 연구실", "🔬 고급 검증"], index=0,
    help="전략 연구실 = 캔들차트 위에 여러 전략을 겹쳐 매수/매도·수익률 비교 / "
         "고급 검증 = 전략 자동 탐색·워크포워드(과최적화 배제) 검증",
)

# 분기에서 공통으로 쓰는 변수 기본값
run, synth_mult, slots = False, None, []
sort_key, train_years, test_years = "CAGR", 3.0, 1.0

st.sidebar.markdown("**종목**")
sig_label = st.sidebar.selectbox("신호 종목 (추세 판단)", list(SIGNAL_TICKERS.keys()), index=0,
                                 help="매수/매도 '신호'를 만드는 기준 종목.")
signal_ticker = SIGNAL_TICKERS[sig_label]
if signal_ticker == "__custom__":
    signal_ticker = st.sidebar.text_input("신호 티커", value="^IXIC").strip().upper()

trd_label = st.sidebar.selectbox("거래 종목 (실제 매매·손익)", list(TRADE_TICKERS.keys()), index=0,
                                 help="실제로 사고파는 종목(차트·손익 계산 대상). 기본은 신호와 동일(1배).")
trade_ticker = TRADE_TICKERS[trd_label]
if trade_ticker == "__synth__":
    synth_mult = st.sidebar.slider("합성 레버리지 배수 (×)", 1.0, 3.0, 3.0, 0.5)
    trade_ticker = f"{signal_ticker}×{synth_mult:g} 합성"
elif trade_ticker == "__same__":
    trade_ticker = signal_ticker
elif trade_ticker == "__custom__":
    trade_ticker = st.sidebar.text_input("거래 티커", value="SOXL").strip().upper()

if view == "📈 전략 연구실":
    mode = "lab"
    tf_label = st.sidebar.selectbox("봉 기준", list(LAB_TF.keys()), index=2,
                                    help="일·주·월봉은 역대 전체기간. 시간봉 약 2년, 분봉 약 7일(야후 제한).")
    interval, ppy, period, intraday = LAB_TF[tf_label]
    fee_bps = st.sidebar.number_input("매매 수수료 (bp)", 0.0, 100.0, 5.0, 1.0)

    st.sidebar.markdown("**차트 표시**")
    ma_periods = st.sidebar.multiselect("이동평균선(MA) 겹쳐보기", [5, 20, 60, 120, 200, 250],
                                        default=[20, 60])
    log_scale = st.sidebar.checkbox("가격 로그 스케일", value=(period == "max"),
                                    help="장기간 비율(%) 비교에 유리. 전체기간에 권장.")

    st.sidebar.markdown("**🎨 겹쳐볼 전략 (최대 4개)**")
    st.sidebar.caption("각 전략을 켜고 파라미터를 세밀하게 조정하세요. 매수 후 보유는 항상 기준선으로 표시됩니다.")
    slots = [
        strategy_slot(0, "추세추종(MA)", True),
        strategy_slot(1, "돌파+트레일", True),
        strategy_slot(2, "SMA교차", False),
        strategy_slot(3, "RSI", False),
    ]
else:
    analysis = st.sidebar.radio("분석 방법", ["🔍 전략 자동 탐색", "🔬 워크포워드 검증"], index=0,
                                help="자동 탐색 = 여러 전략 순위 / 워크포워드 = 과최적화 배제 검증")
    mode = analysis
    tf_label = st.sidebar.selectbox("봉 기준", list(INTERVALS.keys()), index=1)
    interval, ppy, period = INTERVALS[tf_label]
    fee_bps = st.sidebar.number_input("매매 수수료 (bp)", 0.0, 100.0, 5.0, 1.0)
    if mode == "🔬 워크포워드 검증":
        sort_key = st.sidebar.selectbox("최적화 기준(학습 구간)", ["CAGR", "총수익률", "샤프"], index=0)
        train_years = st.sidebar.number_input("학습 기간 (년)", 0.5, 20.0, 3.0, 0.5)
        test_years = st.sidebar.number_input("검증 기간 (년)", 0.25, 5.0, 1.0, 0.25)
    else:
        sort_key = st.sidebar.selectbox("순위 기준", ["CAGR", "총수익률", "샤프"], index=0)
    run = st.sidebar.button("▶ 실행", type="primary", use_container_width=True)


# ============================================================================
# 메인
# ============================================================================
st.title("📈 나스닥 전략 연구실 — 캔들차트 백테스팅")
st.caption("캔들차트 위에 여러 매매 전략을 겹쳐 매수/매도 시점과 수익률을 비교합니다. "
           "교육·연구용이며 투자 권유가 아닙니다.")

with st.expander("📖 사용법 (클릭해서 펼치기)", expanded=False):
    st.markdown(
        """
### 전략 연구실 사용법
1. 왼쪽에서 **신호 종목**(추세 판단 기준)과 **거래 종목**(실제 손익 계산 대상)을 고릅니다. 기본은 둘 다 나스닥 종합(1배)입니다.
2. **봉 기준**을 일/주/월/시간/분봉 중에서 고릅니다. (일·주·월봉은 역대 전체기간)
3. **🎨 전략 1~4** 를 켜고 파라미터를 조정하면, 캔들차트 위에 전략별 **매수 ▲ / 매도 ▼** 시점이 색깔별로 표시되고, 아래에 **전략별 자산곡선**이 함께 그려집니다.
4. 표에서 전략별 **총수익률·CAGR·샤프·MDD·매매횟수**를 한눈에 비교합니다. (기준선 = 매수 후 보유)

> **신호 vs 거래 종목**: 덜 흔들리는 지수(예: ^IXIC)로 *신호*를 만들고, 실제로는 레버리지(SOXL 등)를 *거래*하도록 분리할 수 있습니다. 둘을 같게 두면 한 종목으로 단순 비교가 됩니다.

### 📊 결과 숫자 읽는 법
- **총수익률**: 기간 전체 누적 손익 (+500% = 6배)
- **CAGR**: 연평균 복리 수익률 (기간이 다른 전략을 공정 비교할 때)
- **MDD(최대 낙폭)**: 고점 대비 최대 하락폭 — 0에 가까울수록 안전
- **샤프 지수**: 위험 대비 효율 — 높을수록 좋음(1↑ 양호)

### 🔬 고급 검증 (왼쪽 화면 전환)
- **자동 탐색**: 수십 개 전략 조합을 한 번에 돌려 순위를 매깁니다.
- **워크포워드**: 과거로 전략을 고르고 → '처음 보는' 미래 구간에 적용하길 반복해 **과최적화**를 걸러냅니다.

> ⚠️ 과거 성과가 미래를 보장하지 않습니다. 레버리지 ETF는 매우 위험합니다. 교육·연구용입니다.
        """
    )

with st.expander("📚 용어 사전 (클릭)"):
    st.markdown(
        """
- **총수익률**: 기간 전체 누적 손익. +500% = 원금의 6배.
- **CAGR**: 연평균 복리 수익률 — "매년 평균 몇 %씩 불었나".
- **샤프 지수**: 위험(출렁임) 대비 수익 효율. 1.0=양호, 2.0=훌륭.
- **MDD**: 고점 대비 최대 하락폭. 0에 가까울수록 안전.
- **추세추종(MA)**: 종가가 이동평균선 위면 보유, 아래면 현금.
- **버퍼 밴드**: MA를 살짝(예 5%) 확실히 넘을 때만 매매 → 잦은 헛매매(휩쏘) 감소.
- **돌파+트레일(돈치안+ATR)**: 최근 N봉 고가를 뚫으면 진입, 고점 대비 ATR×k 밀리면 청산(수익은 길게).
- **SMA교차**: 단기 이동평균이 장기 위로 올라오면 매수.
- **RSI**: 과매도에서 시그널선 상향 교차 시 매수, 과매수에서 하향 교차 시 매도.
- **ADX**: 추세 강도. / **이격도**: 현재가가 MA에서 떨어진 정도(과열 판단).
- **합성 레버리지**: 실제 레버리지 ETF가 없던 옛날까지 "지수 × N배"를 시뮬레이션.
- **워크포워드/과최적화**: 안 본 미래 구간으로 실력 확인 → 과거에만 맞춘 전략 걸러내기.
- 종목: **^IXIC**=나스닥종합, **QQQ**=나스닥100, **SOXX**=반도체지수, **SOXL/TQQQ**=3배 레버리지.
        """
    )


# ============================ 전략 연구실 ==================================
if mode == "lab":
    with st.spinner("데이터 불러오는 중..."):
        signal_df = load_data(signal_ticker, interval, period)
        if synth_mult is not None:
            trade_df = synth_leverage_df(signal_df, synth_mult)
        else:
            trade_df = load_data(trade_ticker, interval, period)

    if signal_df.empty or len(signal_df) < 30:
        st.error(f"신호 종목 '{signal_ticker}' 데이터를 충분히 불러오지 못했습니다. "
                 f"티커/봉 기준을 확인하세요. (분봉·시간봉은 야후 제공 기간이 짧습니다)")
        st.stop()
    if trade_df.empty or len(trade_df) < 30:
        st.error(f"거래 종목 '{trade_ticker}' 데이터를 충분히 불러오지 못했습니다.")
        st.stop()

    common_start = max(signal_df.index[0], trade_df.index[0])
    signal_df = signal_df[signal_df.index >= common_start]
    trade_df = trade_df[trade_df.index >= common_start]

    same_tk = signal_ticker == trade_ticker
    hdr = f"{trade_ticker}" if same_tk else f"신호 {signal_ticker} → 거래 {trade_ticker}"
    st.subheader(f"📈 전략 연구실 — {hdr}")
    st.caption(f"기간: {trade_df.index[0].date()} ~ {trade_df.index[-1].date()} "
               f"({len(trade_df):,}개) · 봉: {tf_label}")

    if synth_mult is not None:
        st.info(f"🧪 **합성 레버리지**: '{signal_ticker}' 지수 수익을 {synth_mult:g}배로 시뮬레이션한 "
                f"가상 종목입니다(연 1% 비용 가정). 실제 ETF와 오차가 있을 수 있습니다.")

    # 기준선: 매수 후 보유
    bh_res = run_backtest(trade_df, signal_buy_and_hold(trade_df), 0.0)
    bh_metrics = compute_metrics(bh_res["BuyHoldEquity"], bh_res["DailyReturn"], ppy)
    bh_equity = bh_res["BuyHoldEquity"]

    # 켜진 전략들 백테스트
    strat_results = []
    for i, slot in enumerate(slots):
        if not slot["on"]:
            continue
        spec = slot["spec"]
        pos = map_position(build_position(signal_df, spec), trade_df.index)
        res = run_backtest(trade_df, pos, fee_bps)
        m = compute_metrics(res["StrategyEquity"], res["StrategyReturn"], ppy)
        if not m:
            continue
        strat_results.append({
            "label": f"{i + 1}. {spec_label(spec)}", "color": slot["color"],
            "result": res, "metrics": m, "trades": int((res["Trade"] > 0).sum()),
        })

    if not strat_results:
        st.warning("왼쪽에서 **전략을 최소 1개 켜세요**. (지금은 캔들 + 매수 후 보유 곡선만 표시됩니다)")

    # 요약: 최고 수익 전략 vs 보유
    if strat_results:
        best = max(strat_results, key=lambda s: s["metrics"]["total_return"])
        diff = (best["metrics"]["total_return"] - bh_metrics.get("total_return", 0)) * 100
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("최고 전략", best["label"].split(". ", 1)[-1])
        c2.metric("최고 전략 총수익률", f"{best['metrics']['total_return'] * 100:,.0f}%",
                  f"{diff:+,.0f}%p vs 보유")
        c3.metric("매수 후 보유 총수익률", f"{bh_metrics.get('total_return', 0) * 100:,.0f}%")
        c4.metric("켜진 전략 수", f"{len(strat_results)}개")

    # 차트
    fig = build_lab_fig(trade_df, trade_ticker, ma_periods, log_scale, intraday,
                        strat_results, bh_equity)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 차트 범례를 클릭하면 전략을 끄고 켤 수 있어요. 상단 버튼/하단 슬라이더로 기간 확대. "
               "마커: ▲ 매수 · ▼ 매도 (전략 색마다 살짝 위치를 어긋나게 표시).")

    # 비교 표
    st.markdown("#### 📊 전략별 성과 비교")
    rows, idx = [], []
    for s in strat_results:
        m = s["metrics"]
        rows.append([f"{m['total_return'] * 100:,.0f}%", f"{m.get('cagr', 0) * 100:,.1f}%",
                     f"{m.get('ann_vol', 0) * 100:,.1f}%", f"{m.get('sharpe', 0):,.2f}",
                     f"{m.get('max_dd', 0) * 100:,.0f}%", f"{s['trades']}회"])
        idx.append(s["label"])
    rows.append([f"{bh_metrics.get('total_return', 0) * 100:,.0f}%",
                 f"{bh_metrics.get('cagr', 0) * 100:,.1f}%",
                 f"{bh_metrics.get('ann_vol', 0) * 100:,.1f}%",
                 f"{bh_metrics.get('sharpe', 0):,.2f}",
                 f"{bh_metrics.get('max_dd', 0) * 100:,.0f}%", "—"])
    idx.append("매수 후 보유 (기준)")
    comp = pd.DataFrame(rows, index=idx,
                        columns=["총수익률", "CAGR", "연 변동성", "샤프", "MDD", "매매횟수"])
    st.table(comp)

    # 다운로드: 전략별 자산곡선 합본
    if strat_results:
        dl = pd.DataFrame({"Close": trade_df["Close"], "매수후보유": bh_equity})
        for s in strat_results:
            dl[s["label"]] = s["result"]["StrategyEquity"]
        csv = dl.round(4).to_csv().encode("utf-8-sig")
        st.download_button("⬇ 전략별 자산곡선 CSV 다운로드", csv,
                           file_name=f"{trade_ticker}_strategies.csv", mime="text/csv")
    st.stop()


# ============================ 고급 검증 (실행 버튼) =========================
if not run:
    st.info("👈 왼쪽에서 종목·봉·분석 방법을 고르고 **▶ 실행** 을 누르세요. "
            "(빠른 비교는 위 **📈 전략 연구실** 화면을 이용하세요)")
    st.stop()

with st.spinner("데이터 불러오는 중..."):
    signal_df = load_data(signal_ticker, interval, period)
    if synth_mult is not None:
        trade_df = synth_leverage_df(signal_df, synth_mult)
    else:
        trade_df = load_data(trade_ticker, interval, period)

if signal_df.empty or len(signal_df) < 60:
    st.error(f"신호 종목 '{signal_ticker}' 데이터를 충분히 불러오지 못했습니다.")
    st.stop()
if trade_df.empty or len(trade_df) < 60:
    st.error(f"거래 종목 '{trade_ticker}' 데이터를 충분히 불러오지 못했습니다.")
    st.stop()

common_start = max(signal_df.index[0], trade_df.index[0])
signal_df = signal_df[signal_df.index >= common_start]
trade_df = trade_df[trade_df.index >= common_start]

period_txt = (f"{trade_df.index[0].date()} ~ {trade_df.index[-1].date()} "
              f"({len(trade_df)}개 {tf_label.split()[0]})")
same_tk = signal_ticker == trade_ticker
hdr = f"신호 {signal_ticker} → 거래 {trade_ticker}" if not same_tk else f"{trade_ticker}"

if synth_mult is not None:
    st.info(f"🧪 **합성 레버리지 모드**: '{signal_ticker}' 지수 수익을 {synth_mult:g}배로 "
            f"시뮬레이션(연 1% 비용 가정)한 가상 종목입니다.")

bh_res = run_backtest(trade_df, signal_buy_and_hold(trade_df), 0.0)
bh_metrics = compute_metrics(bh_res["BuyHoldEquity"], bh_res["DailyReturn"], ppy)
bh_total = bh_metrics.get("total_return", 0)


# ---------------------------- 자동 탐색 ------------------------------------
if mode == "🔍 전략 자동 탐색":
    st.subheader(f"🔍 전략 자동 탐색 — {hdr}")
    st.caption(f"기간: {period_txt} · 봉: {tf_label}")
    with st.spinner("여러 전략 조합을 백테스트하는 중..."):
        table = optimize(signal_df, trade_df, ppy, fee_bps, sort_key)
    if table.empty:
        st.error("탐색 결과가 없습니다.")
        st.stop()

    winners = table[table["총수익률"] > bh_total]
    best = table.iloc[0]
    if best["총수익률"] > bh_total:
        st.success(f"✅ '매수 후 보유'({bh_total * 100:,.0f}%)를 이긴 전략 **{len(winners)}개** 발견! "
                   f"최고: **{best['파라미터']}**")
    else:
        st.warning(f"⚠️ 이 종목·기간에서는 보유({bh_total * 100:,.0f}%)를 수익률로 이긴 전략이 없습니다. "
                   f"순위 기준을 '샤프'로 바꿔 위험 대비 효율을 보세요.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("총 수익률", f"{best['총수익률'] * 100:,.0f}%",
              f"{(best['총수익률'] - bh_total) * 100:+,.0f}%p vs 보유")
    c2.metric("CAGR", f"{best['CAGR'] * 100:,.1f}%")
    c3.metric("샤프", f"{best['샤프']:,.2f}")
    c4.metric("MDD", f"{best['MDD'] * 100:,.0f}%")
    c5.metric("매매횟수", f"{best['매매횟수']}")

    st.markdown("#### 전략 순위 (상위 15)")
    show = table.head(15).copy()
    show.insert(0, "순위", range(1, len(show) + 1))
    show["vs 보유"] = (show["총수익률"] - bh_total).map(lambda x: f"{x * 100:+,.0f}%p")
    for col, f in [("총수익률", lambda x: f"{x*100:,.0f}%"), ("CAGR", lambda x: f"{x*100:,.1f}%"),
                   ("샤프", lambda x: f"{x:,.2f}"), ("MDD", lambda x: f"{x*100:,.0f}%")]:
        show[col] = show[col].map(f)
    st.dataframe(show[["순위", "전략종류", "파라미터", "총수익률", "vs 보유", "CAGR", "샤프", "MDD", "매매횟수"]],
                 hide_index=True, use_container_width=True)
    st.caption(f"매수 후 보유: 총수익률 {bh_total*100:,.0f}%, CAGR {bh_metrics.get('cagr',0)*100:,.1f}%, "
               f"MDD {bh_metrics.get('max_dd',0)*100:,.0f}%")

    best_pos = map_position(build_position(signal_df, best["_spec"]), trade_df.index)
    best_res = run_backtest(trade_df, best_pos, fee_bps)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=best_res.index, y=best_res["StrategyEquity"],
                             name="최고 전략", line=dict(color="#2ca02c")))
    fig.add_trace(go.Scatter(x=best_res.index, y=best_res["BuyHoldEquity"],
                             name="매수 후 보유", line=dict(color="#999999", dash="dash")))
    fig.update_layout(height=480, hovermode="x unified", legend=dict(orientation="h"),
                      title=f"최고 전략 vs 보유 ({trade_ticker}, 로그스케일)", yaxis_type="log")
    st.plotly_chart(fig, use_container_width=True)
    st.stop()


# ---------------------------- 워크포워드 검증 ------------------------------
if mode == "🔬 워크포워드 검증":
    st.subheader(f"🔬 워크포워드 검증 — {hdr}")
    st.caption(f"기간: {period_txt} · 봉: {tf_label}")
    train_bars = max(30, int(train_years * ppy))
    test_bars = max(10, int(test_years * ppy))

    if len(trade_df) < train_bars + test_bars:
        st.error(f"데이터가 부족합니다. 현재 {len(trade_df)}개 봉인데 "
                 f"학습({train_bars}) + 검증({test_bars}) = {train_bars + test_bars}개가 필요합니다. "
                 f"학습/검증 기간을 줄이거나, 봉 기준을 일봉/주봉으로 바꿔 보세요.")
        st.stop()

    prog = st.progress(0.0, text="워크포워드 진행 중... (각 구간마다 전략을 새로 최적화)")
    wf_ret, folds = walk_forward(signal_df, trade_df, ppy, fee_bps, sort_key, train_bars, test_bars, prog)
    prog.empty()

    if wf_ret is None or folds is None or folds.empty:
        st.error("검증할 구간을 만들지 못했습니다. 기간 설정을 조정해 주세요.")
        st.stop()

    oos_start, oos_end = wf_ret.index[0], wf_ret.index[-1]
    wf_eq = (1 + wf_ret).cumprod()
    bh_ret_oos = trade_df["Close"].pct_change().fillna(0.0).loc[oos_start:oos_end]
    bh_eq_oos = (1 + bh_ret_oos).cumprod()
    wf_m = compute_metrics(wf_eq, wf_ret, ppy)
    bh_m = compute_metrics(bh_eq_oos, bh_ret_oos, ppy)

    wins = int((folds["초과"] > 0).sum())
    st.markdown(f"**검증(미래) 구간: {oos_start.date()} ~ {oos_end.date()} · 총 {len(folds)}개 구간**")
    if wf_m["total_return"] > bh_m["total_return"]:
        st.success(f"✅ 워크포워드(과최적화 배제) 결과, 전략이 보유를 이겼습니다. "
                   f"{len(folds)}개 검증구간 중 **{wins}개**에서 보유 초과.")
    else:
        st.warning(f"⚠️ 워크포워드 결과, 전략이 보유에 미치지 못했습니다 "
                   f"({len(folds)}개 중 {wins}개 구간만 보유 초과).")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전략 총수익률(검증)", f"{wf_m['total_return'] * 100:,.0f}%",
              f"{(wf_m['total_return'] - bh_m['total_return']) * 100:+,.0f}%p vs 보유")
    c2.metric("전략 CAGR", f"{wf_m['cagr'] * 100:,.1f}%")
    c3.metric("전략 샤프", f"{wf_m['sharpe']:,.2f}", f"보유 {bh_m['sharpe']:,.2f}")
    c4.metric("전략 MDD", f"{wf_m['max_dd'] * 100:,.0f}%", f"보유 {bh_m['max_dd'] * 100:,.0f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wf_eq.index, y=wf_eq, name="워크포워드 전략",
                             line=dict(color="#2ca02c")))
    fig.add_trace(go.Scatter(x=bh_eq_oos.index, y=bh_eq_oos, name="매수 후 보유",
                             line=dict(color="#999999", dash="dash")))
    fig.update_layout(height=460, hovermode="x unified", legend=dict(orientation="h"),
                      title=f"검증 구간 자산곡선 ({trade_ticker}, 로그스케일)", yaxis_type="log")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 구간별 결과 (각 구간은 직전 학습기간에서 고른 전략을 처음 적용한 것)")
    show = folds.copy()
    for col in ("검증수익률", "보유수익률", "초과"):
        show[col] = show[col].map(lambda x: f"{x * 100:+,.0f}%")
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.caption("※ '선택된 전략'이 구간마다 자주 바뀌면 안정적인 단일 전략을 찾기 어렵다는 신호입니다.")
    st.stop()
