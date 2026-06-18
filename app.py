"""
나스닥 백테스팅 웹앱
====================
나스닥(및 미국 주식)의 '역대 전체' 데이터를 주봉 기준으로 불러와
- 단일 전략 백테스트
- '매수 후 보유'를 이기는 전략 자동 탐색(Optimizer)
를 수행하고 결과를 시각화합니다.

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
st.set_page_config(page_title="나스닥 백테스팅", page_icon="📈", layout="wide")

PRESET_TICKERS = {
    "나스닥 종합지수 (^IXIC)": "^IXIC",
    "나스닥100 ETF (QQQ)": "QQQ",
    "S&P500 ETF (SPY)": "SPY",
    "애플 (AAPL)": "AAPL",
    "마이크로소프트 (MSFT)": "MSFT",
    "엔비디아 (NVDA)": "NVDA",
    "테슬라 (TSLA)": "TSLA",
    "아마존 (AMZN)": "AMZN",
    "구글 (GOOGL)": "GOOGL",
    "직접 입력": "__custom__",
}

# 봉 종류 → (yfinance interval, 연간 봉 개수)
INTERVALS = {
    "주봉 (weekly)": ("1wk", 52),
    "일봉 (daily)": ("1d", 252),
    "월봉 (monthly)": ("1mo", 12),
}


# ----------------------------------------------------------------------------
# 데이터 로드
# ----------------------------------------------------------------------------
@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_data(ticker: str, interval: str, use_max: bool, start: dt.date, end: dt.date) -> pd.DataFrame:
    """yfinance 데이터 로드. use_max=True면 역대 전체 기간을 받는다."""
    if use_max:
        df = yf.download(ticker, period="max", interval=interval,
                         auto_adjust=True, progress=False)
    else:
        df = yf.download(ticker, start=start, end=end + dt.timedelta(days=1),
                         interval=interval, auto_adjust=True, progress=False)

    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df


# ----------------------------------------------------------------------------
# 지표 계산
# ----------------------------------------------------------------------------
def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


# ----------------------------------------------------------------------------
# 전략: 매수(1)/현금(0) 포지션 시그널
# ----------------------------------------------------------------------------
def signal_buy_and_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


def signal_sma_crossover(df: pd.DataFrame, short: int, long: int) -> pd.Series:
    short_ma = df["Close"].rolling(short).mean()
    long_ma = df["Close"].rolling(long).mean()
    return (short_ma > long_ma).astype(float)


def signal_rsi(df: pd.DataFrame, period: int, ma_period: int,
               oversold: int, overbought: int) -> pd.Series:
    """과매도(<oversold) 진입 후 RSI가 이동평균선을 상향 돌파 → 매수,
    과매수(>overbought) 진입 후 하향 돌파 → 매도."""
    rsi = compute_rsi(df["Close"], period)
    rsi_ma = rsi.rolling(ma_period).mean()
    cu = ((rsi > rsi_ma) & (rsi.shift(1) <= rsi_ma.shift(1))).to_numpy()
    cd = ((rsi < rsi_ma) & (rsi.shift(1) >= rsi_ma.shift(1))).to_numpy()
    rv = rsi.to_numpy()

    pos = np.full(len(rv), np.nan)
    armed_long = armed_short = False
    for i in range(len(rv)):
        r = rv[i]
        if np.isnan(r):
            continue
        if r < oversold:
            armed_long = True
        if r > overbought:
            armed_short = True
        if armed_long and cu[i]:
            pos[i] = 1.0
            armed_long = False
        elif armed_short and cd[i]:
            pos[i] = 0.0
            armed_short = False
    return pd.Series(pos, index=df.index).ffill().fillna(0.0)


def signal_sma_price(df: pd.DataFrame, window: int) -> pd.Series:
    """종가가 이동평균선 위에 있으면 매수(추세추종)."""
    ma = df["Close"].rolling(window).mean()
    return (df["Close"] > ma).astype(float)


def build_position(df: pd.DataFrame, spec: dict) -> pd.Series:
    """전략 명세(dict)로 포지션 시그널 생성."""
    t = spec["type"]
    if t == "SMA교차":
        return signal_sma_crossover(df, spec["short"], spec["long"])
    if t == "RSI":
        return signal_rsi(df, spec["period"], spec["ma_period"],
                          spec["oversold"], spec["overbought"])
    if t == "추세추종(MA)":
        return signal_sma_price(df, spec["window"])
    return signal_buy_and_hold(df)


def spec_label(spec: dict) -> str:
    t = spec["type"]
    if t == "SMA교차":
        return f"SMA교차 (단기 {spec['short']} / 장기 {spec['long']})"
    if t == "RSI":
        return (f"RSI ({spec['period']}, MA {spec['ma_period']}, "
                f"{spec['oversold']}/{spec['overbought']})")
    if t == "추세추종(MA)":
        return f"추세추종 (MA {spec['window']})"
    return "매수 후 보유"


# ----------------------------------------------------------------------------
# 백테스트 엔진
# ----------------------------------------------------------------------------
def run_backtest(df: pd.DataFrame, position: pd.Series, fee_bps: float = 0.0) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["Close"] = df["Close"]
    out["DailyReturn"] = df["Close"].pct_change().fillna(0.0)

    pos = position.reindex(df.index).fillna(0.0)
    pos_exec = pos.shift(1).fillna(0.0)  # 한 봉 지연 체결(미래참조 방지)
    out["Position"] = pos_exec

    trades = pos_exec.diff().abs().fillna(0.0)
    fee = trades * (fee_bps / 10_000.0)

    out["StrategyReturn"] = out["DailyReturn"] * pos_exec - fee
    out["StrategyEquity"] = (1 + out["StrategyReturn"]).cumprod()
    out["BuyHoldEquity"] = (1 + out["DailyReturn"]).cumprod()
    out["Trade"] = trades
    return out


def compute_metrics(equity: pd.Series, returns: pd.Series, ppy: int) -> dict:
    """ppy: 연간 봉 개수(주봉=52). 연율화에 사용."""
    if equity.empty or len(equity) < 2:
        return {}
    total_return = equity.iloc[-1] - 1
    days = (equity.index[-1] - equity.index[0]).days or 1
    years = days / 365.25
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    ann_vol = returns.std() * np.sqrt(ppy)
    ann_ret = returns.mean() * ppy
    sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan
    running_max = equity.cummax()
    max_dd = (equity / running_max - 1).min()
    return {"total_return": total_return, "cagr": cagr, "ann_vol": ann_vol,
            "sharpe": sharpe, "max_dd": max_dd}


# ----------------------------------------------------------------------------
# 자동 탐색(Optimizer): 보유를 이기는 전략 찾기
# ----------------------------------------------------------------------------
def strategy_grid() -> list:
    """탐색할 전략·파라미터 조합 목록."""
    specs = []
    # 이동평균 교차
    for s in (5, 10, 20, 30):
        for l in (40, 60, 100, 150, 200):
            specs.append({"type": "SMA교차", "short": s, "long": l})
    # 추세추종 (종가 vs 단일 MA)
    for w in (10, 20, 30, 40, 50):
        specs.append({"type": "추세추종(MA)", "window": w})
    # RSI + 시그널선 교차
    for period in (9, 14):
        for ma in (5, 9):
            for ov in (25, 30, 35):
                for ob in (65, 70, 75):
                    specs.append({"type": "RSI", "period": period, "ma_period": ma,
                                  "oversold": ov, "overbought": ob})
    return specs


def optimize(df: pd.DataFrame, ppy: int, fee_bps: float, sort_key: str) -> pd.DataFrame:
    rows = []
    for spec in strategy_grid():
        pos = build_position(df, spec)
        res = run_backtest(df, pos, fee_bps)
        m = compute_metrics(res["StrategyEquity"], res["StrategyReturn"], ppy)
        if not m:
            continue
        rows.append({
            "전략종류": spec["type"],
            "파라미터": spec_label(spec),
            "총수익률": m["total_return"],
            "CAGR": m["cagr"],
            "샤프": m["sharpe"],
            "MDD": m["max_dd"],
            "매매횟수": int((res["Trade"] > 0).sum()),
            "_spec": spec,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(sort_key, ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 사이드바
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ 설정")

mode = st.sidebar.radio("모드", ["🔍 전략 자동 탐색", "📊 단일 전략 백테스트"])

label = st.sidebar.selectbox("종목 선택", list(PRESET_TICKERS.keys()), index=0)
ticker = PRESET_TICKERS[label]
if ticker == "__custom__":
    ticker = st.sidebar.text_input("티커 직접 입력 (예: META)", value="META").strip().upper()

interval_label = st.sidebar.selectbox("봉 기준", list(INTERVALS.keys()), index=0)
interval, ppy = INTERVALS[interval_label]

use_max = st.sidebar.checkbox("역대 전체 기간 사용", value=True)
today = dt.date(2026, 6, 18)
if use_max:
    start_date, end_date = dt.date(1900, 1, 1), today
else:
    c1, c2 = st.sidebar.columns(2)
    start_date = c1.date_input("시작일", value=dt.date(2010, 1, 1))
    end_date = c2.date_input("종료일", value=today)

st.sidebar.markdown("---")
fee_bps = st.sidebar.number_input("매매 수수료 (bp, 1bp=0.01%)",
                                  min_value=0.0, max_value=100.0, value=5.0, step=1.0)

# 모드별 추가 옵션
single_strategy, single_params, sort_key = None, {}, "CAGR"
if mode == "📊 단일 전략 백테스트":
    single_strategy = st.sidebar.selectbox(
        "전략", ["이동평균 교차 (SMA)", "RSI", "추세추종 (MA)", "매수 후 보유 (Buy & Hold)"])
    if single_strategy == "이동평균 교차 (SMA)":
        single_params["short"] = st.sidebar.slider("단기 이동평균", 3, 100, 10)
        single_params["long"] = st.sidebar.slider("장기 이동평균", 10, 300, 40)
    elif single_strategy == "RSI":
        single_params["period"] = st.sidebar.slider("RSI 기간", 5, 30, 14)
        single_params["ma_period"] = st.sidebar.slider("RSI 이동평균선 기간", 2, 30, 9)
        single_params["oversold"] = st.sidebar.slider("과매도 기준(이 아래 진입 후 상향돌파 매수)", 10, 45, 30)
        single_params["overbought"] = st.sidebar.slider("과매수 기준(이 위 진입 후 하향돌파 매도)", 55, 90, 70)
    elif single_strategy == "추세추종 (MA)":
        single_params["window"] = st.sidebar.slider("이동평균 기간", 3, 200, 30)
else:
    sort_key = st.sidebar.selectbox("순위 기준", ["CAGR", "총수익률", "샤프"], index=0)

run = st.sidebar.button("▶ 실행", type="primary", use_container_width=True)


# ----------------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------------
st.title("📈 나스닥 백테스팅")
st.caption("역대 전체 데이터를 주봉 기준으로 백테스트합니다. 교육·연구용이며 투자 권유가 아닙니다.")

if not run:
    st.info("왼쪽에서 모드·종목을 고른 뒤 **실행**을 눌러주세요. "
            "기본값은 '나스닥 종합지수 · 주봉 · 역대 전체 기간 · 전략 자동 탐색' 입니다.")
    st.stop()

if not use_max and start_date >= end_date:
    st.error("시작일이 종료일보다 빨라야 합니다.")
    st.stop()

with st.spinner(f"{ticker} {interval_label} 데이터 불러오는 중..."):
    data = load_data(ticker, interval, use_max, start_date, end_date)

if data.empty or len(data) < 30:
    st.error(f"'{ticker}' 데이터를 충분히 불러오지 못했습니다. 티커/봉 기준을 확인해 주세요.")
    st.stop()

period_txt = f"{data.index[0].date()} ~ {data.index[-1].date()} ({len(data)}개 {interval_label.split()[0]})"
bh_res = run_backtest(data, signal_buy_and_hold(data), fee_bps=0.0)
bh_metrics = compute_metrics(bh_res["BuyHoldEquity"], bh_res["DailyReturn"], ppy)
bh_total = bh_metrics.get("total_return", 0)


# ============================ 자동 탐색 모드 ================================
if mode == "🔍 전략 자동 탐색":
    st.subheader(f"🔍 전략 자동 탐색 — {label.split(' (')[0]}")
    st.caption(f"데이터 기간: {period_txt}")

    with st.spinner("여러 전략·파라미터 조합을 백테스트하는 중..."):
        table = optimize(data, ppy, fee_bps, sort_key)

    if table.empty:
        st.error("탐색 결과가 없습니다.")
        st.stop()

    winners = table[table["총수익률"] > bh_total]
    best = table.iloc[0]

    # 헤드라인
    if best["총수익률"] > bh_total:
        st.success(
            f"✅ '매수 후 보유'({bh_total * 100:,.0f}%)를 이긴 전략 **{len(winners)}개**를 찾았습니다! "
            f"가장 좋은 전략은 **{best['파라미터']}** 입니다."
        )
    else:
        st.warning(
            f"⚠️ 이 종목·기간에서는 '매수 후 보유'({bh_total * 100:,.0f}%)를 "
            f"수익률로 이긴 전략을 찾지 못했습니다. (지수는 장기 우상향이라 보유가 강력합니다.) "
            f"순위 기준을 '샤프'로 바꾸면 위험 대비 효율이 좋은 전략을 볼 수 있습니다."
        )

    # 최고 전략 지표
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("총 수익률", f"{best['총수익률'] * 100:,.0f}%",
              f"{(best['총수익률'] - bh_total) * 100:+,.0f}%p vs 보유")
    c2.metric("CAGR", f"{best['CAGR'] * 100:,.1f}%")
    c3.metric("샤프", f"{best['샤프']:,.2f}")
    c4.metric("MDD", f"{best['MDD'] * 100:,.0f}%")
    c5.metric("매매횟수", f"{best['매매횟수']}")

    # 순위표 (상위 15개 + 보유 비교)
    st.markdown("#### 전략 순위 (상위 15)")
    show = table.head(15).copy()
    show.insert(0, "순위", range(1, len(show) + 1))
    show["vs 보유"] = (show["총수익률"] - bh_total).map(lambda x: f"{x * 100:+,.0f}%p")
    show["총수익률"] = show["총수익률"].map(lambda x: f"{x * 100:,.0f}%")
    show["CAGR"] = show["CAGR"].map(lambda x: f"{x * 100:,.1f}%")
    show["샤프"] = show["샤프"].map(lambda x: f"{x:,.2f}")
    show["MDD"] = show["MDD"].map(lambda x: f"{x * 100:,.0f}%")
    st.dataframe(
        show[["순위", "전략종류", "파라미터", "총수익률", "vs 보유", "CAGR", "샤프", "MDD", "매매횟수"]],
        hide_index=True, use_container_width=True,
    )
    st.caption(f"매수 후 보유 기준: 총수익률 {bh_total * 100:,.0f}%, "
               f"CAGR {bh_metrics.get('cagr', 0) * 100:,.1f}%, "
               f"MDD {bh_metrics.get('max_dd', 0) * 100:,.0f}%")

    # 최고 전략 vs 보유 자산곡선
    best_res = run_backtest(data, build_position(data, best["_spec"]), fee_bps=fee_bps)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=best_res.index, y=best_res["StrategyEquity"],
                             name=f"최고 전략 ({best['전략종류']})", line=dict(color="#2ca02c")))
    fig.add_trace(go.Scatter(x=best_res.index, y=best_res["BuyHoldEquity"],
                             name="매수 후 보유", line=dict(color="#999999", dash="dash")))
    fig.update_layout(height=480, hovermode="x unified", legend=dict(orientation="h"),
                      title="최고 전략 vs 매수 후 보유 (자산 곡선, 시작=1.0)",
                      yaxis_type="log")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("※ 자산 곡선은 로그 스케일입니다(장기 비교에 적합).")
    st.stop()


# ============================ 단일 전략 모드 ================================
st.subheader(f"📊 단일 전략 백테스트 — {label.split(' (')[0]} · {single_strategy}")
st.caption(f"데이터 기간: {period_txt}")

if single_strategy == "이동평균 교차 (SMA)":
    if single_params["short"] >= single_params["long"]:
        st.error("단기 이동평균은 장기보다 작아야 합니다.")
        st.stop()
    position = signal_sma_crossover(data, single_params["short"], single_params["long"])
elif single_strategy == "RSI":
    if single_params["oversold"] >= single_params["overbought"]:
        st.error("과매도 기준은 과매수 기준보다 작아야 합니다.")
        st.stop()
    position = signal_rsi(data, single_params["period"], single_params["ma_period"],
                          single_params["oversold"], single_params["overbought"])
elif single_strategy == "추세추종 (MA)":
    position = signal_sma_price(data, single_params["window"])
else:
    position = signal_buy_and_hold(data)

result = run_backtest(data, position, fee_bps=fee_bps)
metrics = compute_metrics(result["StrategyEquity"], result["StrategyReturn"], ppy)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("총 수익률", f"{metrics.get('total_return', 0) * 100:,.0f}%")
c2.metric("연환산 수익률(CAGR)", f"{metrics.get('cagr', 0) * 100:,.1f}%")
c3.metric("연 변동성", f"{metrics.get('ann_vol', 0) * 100:,.1f}%")
c4.metric("샤프 지수", f"{metrics.get('sharpe', 0):,.2f}")
c5.metric("최대 낙폭(MDD)", f"{metrics.get('max_dd', 0) * 100:,.0f}%")

strat_tr = metrics.get("total_return", 0) * 100
diff = strat_tr - bh_total * 100
st.caption(
    f"같은 기간 매수 후 보유 총 수익률은 **{bh_total * 100:,.0f}%** 입니다. "
    f"이 전략은 그보다 **{diff:+,.0f}%p** {'높습니다 🎉' if diff >= 0 else '낮습니다'}."
)

# 비교표
st.markdown("#### 📊 전략 vs 매수 후 보유 비교")


def _fmt(m):
    return [f"{m.get('total_return', 0) * 100:,.0f}%", f"{m.get('cagr', 0) * 100:,.1f}%",
            f"{m.get('ann_vol', 0) * 100:,.1f}%", f"{m.get('sharpe', 0):,.2f}",
            f"{m.get('max_dd', 0) * 100:,.0f}%"]


compare_df = pd.DataFrame(
    {"내 전략": _fmt(metrics), "매수 후 보유": _fmt(bh_metrics)},
    index=["총 수익률", "연환산 수익률(CAGR)", "연 변동성", "샤프 지수", "최대 낙폭(MDD)"],
)
st.table(compare_df)

# 차트
show_rsi = single_strategy == "RSI"
rows = 3 if show_rsi else 2
heights = [0.45, 0.30, 0.25] if show_rsi else [0.55, 0.45]
titles = (["가격 & 매매 시점", "자산 곡선 (시작=1.0)", "RSI & RSI 이동평균선"]
          if show_rsi else ["가격 & 매매 시점", "자산 곡선 (시작=1.0)"])
fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                    row_heights=heights, subplot_titles=titles)

fig.add_trace(go.Scatter(x=result.index, y=result["Close"], name="종가",
                         line=dict(color="#1f77b4")), row=1, col=1)
pos = result["Position"]
entries = result.index[(pos.diff() > 0)]
exits = result.index[(pos.diff() < 0)]
fig.add_trace(go.Scatter(x=entries, y=result.loc[entries, "Close"], mode="markers", name="매수",
                         marker=dict(symbol="triangle-up", color="green", size=9)), row=1, col=1)
fig.add_trace(go.Scatter(x=exits, y=result.loc[exits, "Close"], mode="markers", name="매도",
                         marker=dict(symbol="triangle-down", color="red", size=9)), row=1, col=1)

fig.add_trace(go.Scatter(x=result.index, y=result["StrategyEquity"], name="전략",
                         line=dict(color="#2ca02c")), row=2, col=1)
fig.add_trace(go.Scatter(x=result.index, y=result["BuyHoldEquity"], name="매수 후 보유",
                         line=dict(color="#999999", dash="dash")), row=2, col=1)

if show_rsi:
    rsi_series = compute_rsi(data["Close"], single_params["period"])
    rsi_ma_series = rsi_series.rolling(single_params["ma_period"]).mean()
    fig.add_trace(go.Scatter(x=result.index, y=rsi_series, name="RSI",
                             line=dict(color="#9467bd")), row=3, col=1)
    fig.add_trace(go.Scatter(x=result.index, y=rsi_ma_series, name="RSI 이동평균",
                             line=dict(color="#ff7f0e", dash="dot")), row=3, col=1)
    fig.add_hline(y=single_params["overbought"], line=dict(color="red", dash="dash"),
                  opacity=0.5, row=3, col=1)
    fig.add_hline(y=single_params["oversold"], line=dict(color="green", dash="dash"),
                  opacity=0.5, row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)

fig.update_layout(height=850 if show_rsi else 700, hovermode="x unified",
                  legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

num_trades = int((result["Trade"] > 0).sum())
st.caption(f"총 매매 횟수: **{num_trades}회** · {period_txt}")
csv = result.round(4).to_csv().encode("utf-8-sig")
st.download_button("⬇ 결과 CSV 다운로드", csv, file_name=f"{ticker}_backtest.csv", mime="text/csv")
