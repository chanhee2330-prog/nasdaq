"""
나스닥/반도체 백테스팅 웹앱
==========================
- 신호 종목(NQ=F, SOXX 등)으로 매매 신호를 만들고
- 실제 손익은 거래 종목(SOXL 등 레버리지 ETF)으로 계산
- 돈치안 돌파 진입 + ATR 트레일링 스톱(수익 길게) + ADX/이격도 필터
- '매수 후 보유'를 이기는 전략 자동 탐색

실행:  streamlit run app.py
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="나스닥/SOXL 백테스팅", page_icon="📈", layout="wide")

# 신호용 종목(추세 판단)
SIGNAL_TICKERS = {
    "나스닥 선물 (NQ=F)": "NQ=F",
    "반도체 지수 ETF (SOXX)": "SOXX",
    "나스닥 종합지수 (^IXIC)": "^IXIC",
    "나스닥100 (QQQ)": "QQQ",
    "직접 입력": "__custom__",
}
# 실제 거래 종목(레버리지 등)
TRADE_TICKERS = {
    "SOXL (반도체 3배 ↑)": "SOXL",
    "TQQQ (나스닥 3배 ↑)": "TQQQ",
    "QQQ (1배)": "QQQ",
    "SOXX (1배)": "SOXX",
    "신호 종목과 동일": "__same__",
    "직접 입력": "__custom__",
}
# 봉 기준 → (interval, 연간 봉 수, 데이터 기간)
INTERVALS = {
    "시간봉 (약 2년)": ("1h", 1700, "730d"),
    "일봉 (전체기간)": ("1d", 252, "max"),
    "주봉 (전체기간)": ("1wk", 52, "max"),
}


# ----------------------------------------------------------------------------
# 데이터
# ----------------------------------------------------------------------------
@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df


# ----------------------------------------------------------------------------
# 지표
# ----------------------------------------------------------------------------
def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    ag = gain.ewm(alpha=1 / period, min_periods=period).mean()
    al = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = ag / al.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, min_periods=n).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    up = h.diff()
    dn = -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1 / n, min_periods=n).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, min_periods=n).mean() / atr_
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, min_periods=n).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1 / n, min_periods=n).mean()


def disparity(close: pd.Series, n: int) -> pd.Series:
    """이격도 = 현재가 / 이동평균 × 100."""
    return close / close.rolling(n).mean() * 100


# ----------------------------------------------------------------------------
# 전략 (신호 종목 기준 포지션 1/0 생성)
# ----------------------------------------------------------------------------
def signal_buy_and_hold(df):
    return pd.Series(1.0, index=df.index)


def signal_sma_crossover(df, short, long):
    return (df["Close"].rolling(short).mean() > df["Close"].rolling(long).mean()).astype(float)


def signal_sma_price(df, window):
    return (df["Close"] > df["Close"].rolling(window).mean()).astype(float)


def signal_rsi(df, period, ma_period, oversold, overbought):
    rsi = compute_rsi(df["Close"], period)
    rma = rsi.rolling(ma_period).mean()
    cu = ((rsi > rma) & (rsi.shift(1) <= rma.shift(1))).to_numpy()
    cd = ((rsi < rma) & (rsi.shift(1) >= rma.shift(1))).to_numpy()
    rv = rsi.to_numpy()
    pos = np.full(len(rv), np.nan)
    al = ash = False
    for i in range(len(rv)):
        r = rv[i]
        if np.isnan(r):
            continue
        if r < oversold:
            al = True
        if r > overbought:
            ash = True
        if al and cu[i]:
            pos[i] = 1.0; al = False
        elif ash and cd[i]:
            pos[i] = 0.0; ash = False
    return pd.Series(pos, index=df.index).ffill().fillna(0.0)


def signal_breakout_trail(df, donchian_n, atr_n, atr_k,
                          use_adx=False, adx_min=20, use_disp=False,
                          disp_cap=115, disp_n=20):
    """돈치안 돌파 진입 + ATR 트레일링 스톱 청산 (+ADX/이격도 진입 필터).
    오르는 동안은 계속 보유하고, 고점 대비 ATR×k 만큼 밀릴 때만 청산."""
    upper = df["High"].rolling(donchian_n).max().shift(1)  # 직전 N봉 고가
    a = atr(df, atr_n)
    adx_v = adx(df) if use_adx else None
    disp_v = disparity(df["Close"], disp_n) if use_disp else None

    c = df["Close"].to_numpy()
    up = upper.to_numpy()
    av = a.to_numpy()
    adxv = adx_v.to_numpy() if use_adx else None
    dv = disp_v.to_numpy() if use_disp else None

    n = len(c)
    pos = np.zeros(n)
    in_pos = False
    peak = 0.0
    for i in range(n):
        if not in_pos:
            enter = (not np.isnan(up[i])) and c[i] > up[i] and not np.isnan(av[i])
            if enter and use_adx and (np.isnan(adxv[i]) or adxv[i] < adx_min):
                enter = False
            if enter and use_disp and (not np.isnan(dv[i]) and dv[i] > disp_cap):
                enter = False
            if enter:
                in_pos = True
                peak = c[i]
                pos[i] = 1.0
        else:
            peak = max(peak, c[i])
            stop = peak - atr_k * av[i] if not np.isnan(av[i]) else -np.inf
            if c[i] < stop:
                in_pos = False
                pos[i] = 0.0
            else:
                pos[i] = 1.0
    return pd.Series(pos, index=df.index)


def build_position(df, spec):
    t = spec["type"]
    if t == "SMA교차":
        return signal_sma_crossover(df, spec["short"], spec["long"])
    if t == "추세추종(MA)":
        return signal_sma_price(df, spec["window"])
    if t == "RSI":
        return signal_rsi(df, spec["period"], spec["ma_period"], spec["oversold"], spec["overbought"])
    if t == "돌파+트레일":
        return signal_breakout_trail(
            df, spec["donchian_n"], spec["atr_n"], spec["atr_k"],
            spec.get("use_adx", False), spec.get("adx_min", 20),
            spec.get("use_disp", False), spec.get("disp_cap", 115), spec.get("disp_n", 20))
    return signal_buy_and_hold(df)


def spec_label(spec):
    t = spec["type"]
    if t == "SMA교차":
        return f"SMA교차(단기{spec['short']}/장기{spec['long']})"
    if t == "추세추종(MA)":
        return f"추세추종(MA{spec['window']})"
    if t == "RSI":
        return f"RSI({spec['period']},MA{spec['ma_period']},{spec['oversold']}/{spec['overbought']})"
    if t == "돌파+트레일":
        flt = []
        if spec.get("use_adx"):
            flt.append(f"ADX≥{spec['adx_min']}")
        if spec.get("use_disp"):
            flt.append(f"이격도≤{spec['disp_cap']}")
        ftxt = (" +" + "/".join(flt)) if flt else ""
        return f"돌파{spec['donchian_n']}+트레일(ATR{spec['atr_n']}×{spec['atr_k']}){ftxt}"
    return "매수 후 보유"


# ----------------------------------------------------------------------------
# 신호 → 거래 종목 정렬 + 백테스트
# ----------------------------------------------------------------------------
def map_position(pos_signal: pd.Series, trade_index: pd.Index) -> pd.Series:
    """신호 종목에서 만든 포지션을 거래 종목 시점에 'as-of'(직전 신호 유지)로 매핑."""
    s = pos_signal[~pos_signal.index.duplicated(keep="last")].sort_index()
    union = s.index.union(trade_index)
    return s.reindex(union).ffill().reindex(trade_index).fillna(0.0)


def run_backtest(trade_df, position, fee_bps=0.0):
    out = pd.DataFrame(index=trade_df.index)
    out["Close"] = trade_df["Close"]
    out["DailyReturn"] = trade_df["Close"].pct_change().fillna(0.0)
    pos = position.reindex(trade_df.index).fillna(0.0)
    pos_exec = pos.shift(1).fillna(0.0)  # 한 봉 지연 체결
    out["Position"] = pos_exec
    trades = pos_exec.diff().abs().fillna(0.0)
    fee = trades * (fee_bps / 10_000.0)
    out["StrategyReturn"] = out["DailyReturn"] * pos_exec - fee
    out["StrategyEquity"] = (1 + out["StrategyReturn"]).cumprod()
    out["BuyHoldEquity"] = (1 + out["DailyReturn"]).cumprod()
    out["Trade"] = trades
    return out


def compute_metrics(equity, returns, ppy):
    if equity.empty or len(equity) < 2:
        return {}
    total_return = equity.iloc[-1] - 1
    days = (equity.index[-1] - equity.index[0]).days or 1
    years = days / 365.25
    cagr = equity.iloc[-1] ** (1 / years) - 1 if (years > 0 and equity.iloc[-1] > 0) else np.nan
    ann_vol = returns.std() * np.sqrt(ppy)
    sharpe = (returns.mean() * ppy) / ann_vol if ann_vol and ann_vol > 0 else np.nan
    max_dd = (equity / equity.cummax() - 1).min()
    return {"total_return": total_return, "cagr": cagr, "ann_vol": ann_vol,
            "sharpe": sharpe, "max_dd": max_dd}


# ----------------------------------------------------------------------------
# 옵티마이저
# ----------------------------------------------------------------------------
def strategy_grid():
    specs = []
    for dn in (20, 40, 55):
        for k in (2.0, 3.0):
            specs.append({"type": "돌파+트레일", "donchian_n": dn, "atr_n": 14, "atr_k": k})
            specs.append({"type": "돌파+트레일", "donchian_n": dn, "atr_n": 14, "atr_k": k,
                          "use_adx": True, "adx_min": 20})
    for w in (20, 50, 100, 200):
        specs.append({"type": "추세추종(MA)", "window": w})
    for s in (10, 20, 30):
        for l in (50, 100, 200):
            specs.append({"type": "SMA교차", "short": s, "long": l})
    for (p, m, ov, ob) in [(14, 9, 30, 70), (14, 5, 35, 65)]:
        specs.append({"type": "RSI", "period": p, "ma_period": m, "oversold": ov, "overbought": ob})
    return specs


def optimize(signal_df, trade_df, ppy, fee_bps, sort_key):
    rows = []
    for spec in strategy_grid():
        pos_sig = build_position(signal_df, spec)
        pos = map_position(pos_sig, trade_df.index)
        res = run_backtest(trade_df, pos, fee_bps)
        m = compute_metrics(res["StrategyEquity"], res["StrategyReturn"], ppy)
        if not m:
            continue
        rows.append({"전략종류": spec["type"], "파라미터": spec_label(spec),
                     "총수익률": m["total_return"], "CAGR": m["cagr"], "샤프": m["sharpe"],
                     "MDD": m["max_dd"], "매매횟수": int((res["Trade"] > 0).sum()), "_spec": spec})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(sort_key, ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 사이드바
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ 설정")
mode = st.sidebar.radio("모드", ["🔍 전략 자동 탐색", "📊 단일 전략 백테스트"])

st.sidebar.markdown("**종목**")
sig_label = st.sidebar.selectbox("신호 종목 (추세 판단)", list(SIGNAL_TICKERS.keys()), index=0)
signal_ticker = SIGNAL_TICKERS[sig_label]
if signal_ticker == "__custom__":
    signal_ticker = st.sidebar.text_input("신호 티커", value="NQ=F").strip().upper()

trd_label = st.sidebar.selectbox("거래 종목 (실제 매매)", list(TRADE_TICKERS.keys()), index=0)
trade_ticker = TRADE_TICKERS[trd_label]
if trade_ticker == "__same__":
    trade_ticker = signal_ticker
elif trade_ticker == "__custom__":
    trade_ticker = st.sidebar.text_input("거래 티커", value="SOXL").strip().upper()

interval_label = st.sidebar.selectbox("봉 기준", list(INTERVALS.keys()), index=1)
interval, ppy, period = INTERVALS[interval_label]

st.sidebar.markdown("---")
fee_bps = st.sidebar.number_input("매매 수수료 (bp)", 0.0, 100.0, 5.0, 1.0)

single_strategy, sp, sort_key = None, {}, "CAGR"
if mode == "📊 단일 전략 백테스트":
    single_strategy = st.sidebar.selectbox(
        "전략", ["돌파+트레일 (추천)", "추세추종 (MA)", "이동평균 교차 (SMA)", "RSI", "매수 후 보유"])
    if single_strategy == "돌파+트레일 (추천)":
        sp["donchian_n"] = st.sidebar.slider("돈치안 돌파 기간(진입)", 5, 100, 20)
        sp["atr_n"] = st.sidebar.slider("ATR 기간", 5, 30, 14)
        sp["atr_k"] = st.sidebar.slider("ATR 트레일링 배수(클수록 길게 보유)", 1.0, 6.0, 3.0, 0.5)
        sp["use_adx"] = st.sidebar.checkbox("ADX 추세강도 필터", value=False)
        if sp["use_adx"]:
            sp["adx_min"] = st.sidebar.slider("최소 ADX", 10, 40, 20)
        sp["use_disp"] = st.sidebar.checkbox("이격도 과열 필터", value=False)
        if sp["use_disp"]:
            sp["disp_cap"] = st.sidebar.slider("이격도 상한(이 위면 진입 금지)", 101, 140, 115)
            sp["disp_n"] = st.sidebar.slider("이격도 이동평균 기간", 5, 60, 20)
    elif single_strategy == "추세추종 (MA)":
        sp["window"] = st.sidebar.slider("이동평균 기간", 3, 200, 50)
    elif single_strategy == "이동평균 교차 (SMA)":
        sp["short"] = st.sidebar.slider("단기", 3, 100, 20)
        sp["long"] = st.sidebar.slider("장기", 10, 300, 100)
    elif single_strategy == "RSI":
        sp["period"] = st.sidebar.slider("RSI 기간", 5, 30, 14)
        sp["ma_period"] = st.sidebar.slider("RSI 이동평균", 2, 30, 9)
        sp["oversold"] = st.sidebar.slider("과매도", 10, 45, 30)
        sp["overbought"] = st.sidebar.slider("과매수", 55, 90, 70)
else:
    sort_key = st.sidebar.selectbox("순위 기준", ["CAGR", "총수익률", "샤프"], index=0)

run = st.sidebar.button("▶ 실행", type="primary", use_container_width=True)


# ----------------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------------
st.title("📈 나스닥/SOXL 백테스팅")
st.caption("신호는 기초자산(NQ=F·SOXX 등)으로 만들고, 손익은 거래 종목(SOXL 등)으로 계산합니다. "
           "교육·연구용이며 투자 권유가 아닙니다. ⚠️ 레버리지 ETF는 횡보·하락장에서 가치가 크게 감소합니다.")

if not run:
    st.info("왼쪽에서 신호 종목·거래 종목·봉 기준을 고른 뒤 **실행**을 눌러주세요. "
            "기본값은 'NQ=F 신호 → SOXL 거래 · 일봉 전체기간 · 전략 자동 탐색' 입니다.")
    st.stop()

spin = st.spinner("데이터 불러오는 중...")
with spin:
    signal_df = load_data(signal_ticker, interval, period)
    trade_df = load_data(trade_ticker, interval, period)

if signal_df.empty or len(signal_df) < 60:
    st.error(f"신호 종목 '{signal_ticker}' 데이터를 충분히 불러오지 못했습니다.")
    st.stop()
if trade_df.empty or len(trade_df) < 60:
    st.error(f"거래 종목 '{trade_ticker}' 데이터를 충분히 불러오지 못했습니다.")
    st.stop()

# 신호·거래 기간 겹치는 구간으로 제한
common_start = max(signal_df.index[0], trade_df.index[0])
signal_df = signal_df[signal_df.index >= common_start]
trade_df = trade_df[trade_df.index >= common_start]

period_txt = (f"{trade_df.index[0].date()} ~ {trade_df.index[-1].date()} "
              f"({len(trade_df)}개 {interval_label.split()[0]})")
same_tk = signal_ticker == trade_ticker
hdr = f"신호 {signal_ticker} → 거래 {trade_ticker}" if not same_tk else f"{trade_ticker}"

bh_res = run_backtest(trade_df, signal_buy_and_hold(trade_df), 0.0)
bh_metrics = compute_metrics(bh_res["BuyHoldEquity"], bh_res["DailyReturn"], ppy)
bh_total = bh_metrics.get("total_return", 0)


# ============================ 자동 탐색 ====================================
if mode == "🔍 전략 자동 탐색":
    st.subheader(f"🔍 전략 자동 탐색 — {hdr}")
    st.caption(f"기간: {period_txt} · 봉: {interval_label}")
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
                      title=f"최고 전략 vs 보유 ({trade_ticker}, 자산곡선 로그스케일)",
                      yaxis_type="log")
    st.plotly_chart(fig, use_container_width=True)
    st.stop()


# ============================ 단일 전략 ====================================
st.subheader(f"📊 단일 전략 — {hdr} · {single_strategy}")
st.caption(f"기간: {period_txt} · 봉: {interval_label}")

if single_strategy == "이동평균 교차 (SMA)" and sp["short"] >= sp["long"]:
    st.error("단기 이동평균은 장기보다 작아야 합니다.")
    st.stop()
if single_strategy == "RSI" and sp["oversold"] >= sp["overbought"]:
    st.error("과매도 기준은 과매수 기준보다 작아야 합니다.")
    st.stop()

spec_map = {
    "돌파+트레일 (추천)": {"type": "돌파+트레일", "atr_n": sp.get("atr_n", 14), **sp},
    "추세추종 (MA)": {"type": "추세추종(MA)", **sp},
    "이동평균 교차 (SMA)": {"type": "SMA교차", **sp},
    "RSI": {"type": "RSI", **sp},
    "매수 후 보유": {"type": "BH"},
}
spec = spec_map[single_strategy]
position = map_position(build_position(signal_df, spec), trade_df.index)
result = run_backtest(trade_df, position, fee_bps)
metrics = compute_metrics(result["StrategyEquity"], result["StrategyReturn"], ppy)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("총 수익률", f"{metrics.get('total_return', 0) * 100:,.0f}%")
c2.metric("CAGR", f"{metrics.get('cagr', 0) * 100:,.1f}%")
c3.metric("연 변동성", f"{metrics.get('ann_vol', 0) * 100:,.1f}%")
c4.metric("샤프", f"{metrics.get('sharpe', 0):,.2f}")
c5.metric("MDD", f"{metrics.get('max_dd', 0) * 100:,.0f}%")

diff = (metrics.get("total_return", 0) - bh_total) * 100
st.caption(f"같은 기간 {trade_ticker} 매수 후 보유 총수익률은 **{bh_total*100:,.0f}%**. "
           f"이 전략은 **{diff:+,.0f}%p** {'높습니다 🎉' if diff >= 0 else '낮습니다'}.")

st.markdown("#### 📊 전략 vs 매수 후 보유")


def _fmt(m):
    return [f"{m.get('total_return',0)*100:,.0f}%", f"{m.get('cagr',0)*100:,.1f}%",
            f"{m.get('ann_vol',0)*100:,.1f}%", f"{m.get('sharpe',0):,.2f}",
            f"{m.get('max_dd',0)*100:,.0f}%"]


st.table(pd.DataFrame({"내 전략": _fmt(metrics), "매수 후 보유": _fmt(bh_metrics)},
                      index=["총 수익률", "CAGR", "연 변동성", "샤프", "MDD"]))

# 차트
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                    row_heights=[0.55, 0.45],
                    subplot_titles=(f"{trade_ticker} 가격 & 매매 시점", "자산 곡선 (로그스케일)"))
fig.add_trace(go.Scatter(x=result.index, y=result["Close"], name=trade_ticker,
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
fig.update_yaxes(type="log", row=2, col=1)
fig.update_layout(height=720, hovermode="x unified", legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

num_trades = int((result["Trade"] > 0).sum())
st.caption(f"총 매매 횟수: **{num_trades}회** · {period_txt}")
csv = result.round(4).to_csv().encode("utf-8-sig")
st.download_button("⬇ 결과 CSV 다운로드", csv, file_name=f"{trade_ticker}_backtest.csv", mime="text/csv")
