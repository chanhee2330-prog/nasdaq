"""
나스닥 백테스팅 웹앱
====================
나스닥(및 미국 주식) 종목의 과거 데이터를 불러와
여러 가지 기본 매매 전략을 백테스트하고 결과를 시각화합니다.

실행:  streamlit run app.py
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------------
# 페이지 설정
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="나스닥 백테스팅",
    page_icon="📈",
    layout="wide",
)

# 자주 쓰는 나스닥 관련 종목 (라벨 → 티커)
PRESET_TICKERS = {
    "나스닥 종합지수 (^IXIC)": "^IXIC",
    "나스닥100 ETF (QQQ)": "QQQ",
    "애플 (AAPL)": "AAPL",
    "마이크로소프트 (MSFT)": "MSFT",
    "엔비디아 (NVDA)": "NVDA",
    "테슬라 (TSLA)": "TSLA",
    "아마존 (AMZN)": "AMZN",
    "구글 (GOOGL)": "GOOGL",
    "직접 입력": "__custom__",
}

TRADING_DAYS = 252  # 연간 거래일 수 (지표 연율화에 사용)


# ----------------------------------------------------------------------------
# 데이터 로드
# ----------------------------------------------------------------------------
@st.cache_data(ttl=60 * 60)  # 1시간 캐시
def load_data(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """yfinance로 일봉 데이터를 받아 종가(Close) 기준 DataFrame 반환."""
    df = yf.download(
        ticker,
        start=start,
        end=end + dt.timedelta(days=1),  # end 당일 포함
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance가 MultiIndex 컬럼을 줄 때가 있어 평탄화
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df


# ----------------------------------------------------------------------------
# 전략: 매수(1)/현금(0) 포지션 시그널 생성
# ----------------------------------------------------------------------------
def signal_buy_and_hold(df: pd.DataFrame) -> pd.Series:
    """항상 보유 (벤치마크)."""
    return pd.Series(1.0, index=df.index)


def signal_sma_crossover(df: pd.DataFrame, short: int, long: int) -> pd.Series:
    """단기 이동평균이 장기 이동평균 위에 있으면 매수."""
    short_ma = df["Close"].rolling(short).mean()
    long_ma = df["Close"].rolling(long).mean()
    return (short_ma > long_ma).astype(float)


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """RSI(상대강도지수) 계산."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def signal_rsi(df: pd.DataFrame, period: int, oversold: int, overbought: int) -> pd.Series:
    """RSI가 과매도선 아래로 가면 매수, 과매수선 위로 가면 매도(현금)."""
    rsi = compute_rsi(df["Close"], period)
    pos = pd.Series(np.nan, index=df.index)
    pos[rsi < oversold] = 1.0   # 과매도 → 매수
    pos[rsi > overbought] = 0.0  # 과매수 → 청산
    pos = pos.ffill().fillna(0.0)
    return pos


# ----------------------------------------------------------------------------
# 백테스트 엔진
# ----------------------------------------------------------------------------
def run_backtest(df: pd.DataFrame, position: pd.Series, fee_bps: float = 0.0) -> pd.DataFrame:
    """
    포지션 시그널로 일별 수익률을 계산.
    당일 종가에 시그널이 정해지면 '다음 날'부터 반영(미래참조 방지).
    fee_bps: 매매 1회당 수수료(베이시스포인트, 1bp=0.01%).
    """
    out = pd.DataFrame(index=df.index)
    out["Close"] = df["Close"]
    out["DailyReturn"] = df["Close"].pct_change().fillna(0.0)

    pos = position.reindex(df.index).fillna(0.0)
    pos_exec = pos.shift(1).fillna(0.0)  # 하루 지연 체결
    out["Position"] = pos_exec

    # 포지션이 바뀌는 날 거래비용 차감
    trades = pos_exec.diff().abs().fillna(0.0)
    fee = trades * (fee_bps / 10_000.0)

    out["StrategyReturn"] = out["DailyReturn"] * pos_exec - fee
    out["StrategyEquity"] = (1 + out["StrategyReturn"]).cumprod()
    out["BuyHoldEquity"] = (1 + out["DailyReturn"]).cumprod()
    out["Trade"] = trades
    return out


def compute_metrics(equity: pd.Series, returns: pd.Series) -> dict:
    """누적수익률, 연환산수익률(CAGR), 변동성, 샤프, 최대낙폭 등."""
    if equity.empty or len(equity) < 2:
        return {}

    total_return = equity.iloc[-1] - 1
    days = (equity.index[-1] - equity.index[0]).days or 1
    years = days / 365.25
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan

    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)
    ann_ret = returns.mean() * TRADING_DAYS
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()

    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
    }


# ----------------------------------------------------------------------------
# 사이드바 (입력)
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ 설정")

label = st.sidebar.selectbox("종목 선택", list(PRESET_TICKERS.keys()), index=1)
ticker = PRESET_TICKERS[label]
if ticker == "__custom__":
    ticker = st.sidebar.text_input("티커 직접 입력 (예: META)", value="META").strip().upper()

today = dt.date(2026, 6, 18)
col_a, col_b = st.sidebar.columns(2)
start_date = col_a.date_input("시작일", value=dt.date(2018, 1, 1))
end_date = col_b.date_input("종료일", value=today)

st.sidebar.markdown("---")
strategy = st.sidebar.selectbox(
    "전략",
    ["이동평균 교차 (SMA)", "RSI", "매수 후 보유 (Buy & Hold)"],
)

# 전략별 파라미터
params = {}
if strategy == "이동평균 교차 (SMA)":
    params["short"] = st.sidebar.slider("단기 이동평균 (일)", 5, 100, 20)
    params["long"] = st.sidebar.slider("장기 이동평균 (일)", 20, 300, 60)
elif strategy == "RSI":
    params["period"] = st.sidebar.slider("RSI 기간 (일)", 5, 30, 14)
    params["oversold"] = st.sidebar.slider("과매도 기준 (이하면 매수)", 10, 45, 30)
    params["overbought"] = st.sidebar.slider("과매수 기준 (이상이면 매도)", 55, 90, 70)

st.sidebar.markdown("---")
fee_bps = st.sidebar.number_input(
    "매매 수수료 (bp, 1bp=0.01%)", min_value=0.0, max_value=100.0, value=5.0, step=1.0
)

run = st.sidebar.button("▶ 백테스트 실행", type="primary", use_container_width=True)


# ----------------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------------
st.title("📈 나스닥 백테스팅")
st.caption("과거 데이터로 매매 전략을 검증해 보세요. 투자 권유가 아니며 교육·연구용입니다.")

if not run:
    st.info("왼쪽 사이드바에서 종목·기간·전략을 고른 뒤 **백테스트 실행**을 눌러주세요.")
    st.stop()

if start_date >= end_date:
    st.error("시작일이 종료일보다 빠르거나 같아야 합니다.")
    st.stop()

with st.spinner(f"{ticker} 데이터 불러오는 중..."):
    data = load_data(ticker, start_date, end_date)

if data.empty:
    st.error(f"'{ticker}' 데이터를 불러오지 못했습니다. 티커나 기간을 확인해 주세요.")
    st.stop()

# 전략 시그널 생성
if strategy == "이동평균 교차 (SMA)":
    if params["short"] >= params["long"]:
        st.error("단기 이동평균은 장기 이동평균보다 작아야 합니다.")
        st.stop()
    position = signal_sma_crossover(data, params["short"], params["long"])
elif strategy == "RSI":
    if params["oversold"] >= params["overbought"]:
        st.error("과매도 기준은 과매수 기준보다 작아야 합니다.")
        st.stop()
    position = signal_rsi(data, params["period"], params["oversold"], params["overbought"])
else:
    position = signal_buy_and_hold(data)

result = run_backtest(data, position, fee_bps=fee_bps)
metrics = compute_metrics(result["StrategyEquity"], result["StrategyReturn"])
bh_metrics = compute_metrics(result["BuyHoldEquity"], result["DailyReturn"])

# ----- 핵심 지표 -----
st.subheader(f"결과 요약 — {label.split(' (')[0]} · {strategy}")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("총 수익률", f"{metrics.get('total_return', 0) * 100:,.1f}%")
c2.metric("연환산 수익률(CAGR)", f"{metrics.get('cagr', 0) * 100:,.1f}%")
c3.metric("연 변동성", f"{metrics.get('ann_vol', 0) * 100:,.1f}%")
c4.metric("샤프 지수", f"{metrics.get('sharpe', 0):,.2f}")
c5.metric("최대 낙폭(MDD)", f"{metrics.get('max_dd', 0) * 100:,.1f}%")

bh_tr = bh_metrics.get("total_return", 0) * 100
strat_tr = metrics.get("total_return", 0) * 100
diff = strat_tr - bh_tr
st.caption(
    f"같은 기간 매수 후 보유(Buy & Hold) 총 수익률은 **{bh_tr:,.1f}%** 입니다. "
    f"이 전략은 그보다 **{diff:+,.1f}%p** {'높습니다 🎉' if diff >= 0 else '낮습니다'}."
)

# ----- 차트: 가격 + 자산곡선 -----
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
    row_heights=[0.55, 0.45],
    subplot_titles=("가격 & 매매 시점", "자산 곡선 (시작=1.0)"),
)

# 가격
fig.add_trace(
    go.Scatter(x=result.index, y=result["Close"], name="종가", line=dict(color="#1f77b4")),
    row=1, col=1,
)

# 매매 시점 마커 (포지션이 0→1 진입, 1→0 청산)
pos = result["Position"]
entries = result.index[(pos.diff() > 0)]
exits = result.index[(pos.diff() < 0)]
fig.add_trace(
    go.Scatter(x=entries, y=result.loc[entries, "Close"], mode="markers", name="매수",
               marker=dict(symbol="triangle-up", color="green", size=10)),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(x=exits, y=result.loc[exits, "Close"], mode="markers", name="매도",
               marker=dict(symbol="triangle-down", color="red", size=10)),
    row=1, col=1,
)

# 자산곡선 (전략 vs 매수후보유)
fig.add_trace(
    go.Scatter(x=result.index, y=result["StrategyEquity"], name="전략",
               line=dict(color="#2ca02c")),
    row=2, col=1,
)
fig.add_trace(
    go.Scatter(x=result.index, y=result["BuyHoldEquity"], name="매수 후 보유",
               line=dict(color="#999999", dash="dash")),
    row=2, col=1,
)

fig.update_layout(height=700, hovermode="x unified", legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

# ----- 거래 횟수 & 데이터 다운로드 -----
num_trades = int((result["Trade"] > 0).sum())
st.caption(f"총 매매 횟수: **{num_trades}회** · 데이터 기간: {result.index[0].date()} ~ {result.index[-1].date()}")

csv = result.round(4).to_csv().encode("utf-8-sig")
st.download_button("⬇ 결과 CSV 다운로드", csv, file_name=f"{ticker}_backtest.csv", mime="text/csv")

with st.expander("📋 원본 데이터 미리보기"):
    st.dataframe(result.tail(30))
