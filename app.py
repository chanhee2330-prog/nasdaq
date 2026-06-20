"""
SOXL 추세추종 신호기 (단일 전략)
================================
검증으로 고른 '최고의 전략'만 남긴 버전.
- 신호: SOXX(반도체 지수)의 이동평균(MA150~250) ± 버퍼
- 매매: SOXX 종가가 MA×(1+버퍼) 위면 SOXL(3배) 보유, MA×(1−버퍼) 아래면 현금
- '오늘의 신호' + 자산곡선 + 비중관리까지 한 화면에.

실행:  streamlit run app.py
"""

import warnings

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", message="Unverified HTTPS request")


@st.cache_resource(show_spinner=False)
def _yf_session():
    try:
        from curl_cffi import requests as _creq
        return _creq.Session(impersonate="chrome", verify=False)
    except Exception:
        return None


from engine import (
    build_position, compute_metrics, map_position, run_backtest,
    signal_buy_and_hold, signal_rsi_channel, signal_trend_adaptive,
    synth_leverage_df, trend_adaptive_lines,
)

st.set_page_config(page_title="SOXL 추세추종 신호기", page_icon="📈", layout="wide")

SIGNAL_TICKERS = {
    "반도체 지수 SOXX (권장)": "SOXX",
    "나스닥100 QQQ": "QQQ",
    "나스닥 종합 ^IXIC (역대 최장)": "^IXIC",
}
TRADE_TICKERS = {
    "SOXL (반도체 3배)": "SOXL",
    "TQQQ (나스닥 3배)": "TQQQ",
    "합성 3배 (신호지수 ×3, 전체기간)": "__synth__",
}
UP, DOWN = "#d62728", "#1f77b4"
CHART_CONFIG = {"scrollZoom": True, "displaylogo": False, "doubleClick": "reset"}


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="max", interval="1d",
                     auto_adjust=True, progress=False, session=_yf_session())
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df


def _memo_rsi_trend(df):
    """메모(영상) 전략: RSI 과매도/과매수 채널 + 200일선 위에서만 보유(롱only)."""
    rsi_pos = signal_rsi_channel(df, period=14, oversold=30, overbought=70)
    ma200 = df["Close"].rolling(200).mean()
    return rsi_pos.astype(float) * (df["Close"] > ma200).astype(float)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def compare_strategies(df, fee_bps, my_spec):
    """같은 종목·기간·수수료로 내 전략 vs 메모 RSI 전략 vs 보유를 비교.
    반환: (지표표 DataFrame, {전략명: 자산곡선 Series})."""
    items = [
        ("① 정밀추세 (내 전략)", build_position(df, my_spec), fee_bps),
        ("② 단순 RSI 30/70", signal_rsi_channel(df, 14, 30, 70), fee_bps),
        ("③ RSI+200MA 롱only (메모)", _memo_rsi_trend(df), fee_bps),
        ("④ 그냥 보유", signal_buy_and_hold(df), 0.0),
    ]
    rows, curves = [], {}
    for name, pos, fee in items:
        res = run_backtest(df, pos, fee)
        m = compute_metrics(res["StrategyEquity"], res["StrategyReturn"], 252)
        rows.append({"전략": name, "총수익률": m.get("total_return", 0), "CAGR": m.get("cagr", 0),
                     "MDD": m.get("max_dd", 0), "샤프": m.get("sharpe", 0),
                     "거래수": int((res["Trade"] > 0).sum())})
        curves[name] = res["StrategyEquity"]
    return pd.DataFrame(rows), curves


# ----------------------------------------------------------------------------
# 사이드바
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ 설정")
sig_label = st.sidebar.selectbox("신호 종목 (추세 판단)", list(SIGNAL_TICKERS.keys()), index=0,
                                 help="1배 지수로 추세를 판단합니다. 3배 ETF보다 신호가 깨끗해요.")
signal_ticker = SIGNAL_TICKERS[sig_label]

trd_label = st.sidebar.selectbox("거래 종목 (실제 매매·손익)", list(TRADE_TICKERS.keys()), index=0)
trade_code = TRADE_TICKERS[trd_label]

ma_n = st.sidebar.slider("이동평균 기간 (일)", 100, 250, 150, 10,
                         help="추세의 중심선. 종가가 이 MA 위/아래인지로 추세를 판단. 150~200 권장.")
use_ema = st.sidebar.checkbox("EMA 사용 (폭락 더 빨리 감지)", value=False,
                              help="SMA 대신 EMA(지수이동평균)를 쓰면 최근 가격에 민감해 폭락을 몇 봉 더 일찍 회피합니다. "
                                   "대신 가짜하락에도 조금 더 민감해질 수 있어요.")

st.sidebar.markdown("**🔬 가짜하락 정밀 감지**")
k_exit = st.sidebar.slider("청산 밴드 (×ATR)", 0.5, 4.0, 1.5, 0.5,
                           help="청산선 = MA − (이 값)×ATR. ATR(변동성)에 비례하므로 변동성 큰 장에선 밴드가 "
                                "자동으로 넓어져 '평소 출렁임=가짜하락'에 안 털립니다. 클수록 더 잘 버팀(둔감).")
confirm_bars = st.sidebar.slider("지속 확인 (봉)", 1, 5, 2, 1,
                                 help="청산선을 이 봉 수만큼 '연속' 깨야 진짜 하락으로 인정 → 하루짜리 가짜하락 무시. "
                                      "클수록 가짜에 덜 속지만 진짜 청산이 늦어집니다.")
slope_n = st.sidebar.slider("추세 방향 판단 기간 (봉)", 5, 40, 10, 5,
                            help="MA가 이 기간 전보다 위면 '상승 추세'. 상승 중이면 밴드 밑 하락도 '눌림(가짜)'으로 보고 "
                                 "절반만 축소, MA가 꺾였으면 '진짜 붕괴'로 보고 전량 청산합니다.")
k_enter = st.sidebar.slider("진입 강도 (×ATR)", 0.0, 3.0, 0.5, 0.5,
                            help="진입선 = MA + (이 값)×ATR. 이만큼 확실히 올라설 때만 신규 진입(가짜 반등 매수 방지).")
cooldown = st.sidebar.slider("재진입 쿨다운 (봉)", 0, 20, 3, 1,
                             help="청산 직후 이 봉 수 동안 재매수 금지 → 천장 부근 반복 진입(churn) 억제.")
fee_bps = st.sidebar.number_input("매매 수수료 (bp)", 0.0, 100.0, 5.0, 1.0)
weight = st.sidebar.slider("포트폴리오 중 3배 비중 (%)", 10, 100, 50, 10,
                           help="3배 ETF에 넣는 비중. 낮출수록 내 계좌 낙폭이 줄어요.") / 100.0
show_vol = st.sidebar.checkbox("거래량 폭증(투매 바닥) 표시", value=True,
                               help="거래량이 20일 평균보다 폭증한 하락일 = 단기 바닥 후보를 차트에 ★로 표시. "
                                    "검증상 5~10일 단기 반등에 약한 엣지(승률~58%). 추세 매매의 '눌림 재진입 타이밍' 참고용.")
vol_mult = st.sidebar.slider("거래량 폭증 기준 (×20일평균)", 1.5, 4.0, 2.0, 0.5) if show_vol else 2.0


# ----------------------------------------------------------------------------
# 데이터 & 백테스트
# ----------------------------------------------------------------------------
st.title("📈 SOXL 정밀 추세추종 신호기")
ma_kind = "EMA" if use_ema else "MA"
st.caption(f"**{signal_ticker}의 {ma_kind}{ma_n}선 기준 — 추세면 {trd_label.split(' ')[0]} 보유, "
           f"진짜 폭락이면 단계적으로 빠집니다.** 가짜하락(변동성·하루짜리 흔들기)은 ATR밴드·지속확인·"
           f"MA기울기로 걸러 '절반만 축소'하고, 진짜 추세붕괴만 전량 청산. 교육·연구용.")

with st.spinner("데이터 불러오는 중..."):
    signal_df = load_data(signal_ticker)
    if trade_code == "__synth__":
        trade_df = synth_leverage_df(signal_df, 3.0)
        trade_name = f"{signal_ticker}×3 합성"
    else:
        trade_df = load_data(trade_code)
        trade_name = trade_code

if signal_df.empty or len(signal_df) < ma_n + 10:
    st.error(f"신호 종목 '{signal_ticker}' 데이터를 충분히 불러오지 못했습니다.")
    st.stop()
if trade_df.empty or len(trade_df) < 60:
    st.error(f"거래 종목 '{trade_name}' 데이터를 불러오지 못했습니다.")
    st.stop()

common = max(signal_df.index[0], trade_df.index[0])
signal_df = signal_df[signal_df.index >= common]
trade_df = trade_df[trade_df.index >= common]

ma, upper, lower = trend_adaptive_lines(signal_df, ma_n, use_ema, 14, k_enter, k_exit)

spec = {"type": "정밀추세", "window": ma_n, "ema": use_ema, "atr_n": 14,
        "k_enter": k_enter, "k_exit": k_exit, "confirm_bars": confirm_bars,
        "slope_n": slope_n, "cooldown": cooldown}
position = map_position(build_position(signal_df, spec), trade_df.index)
result = run_backtest(trade_df, position, fee_bps)
metrics = compute_metrics(result["StrategyEquity"], result["StrategyReturn"], 252)

bh_res = run_backtest(trade_df, signal_buy_and_hold(trade_df), 0.0)
bh_metrics = compute_metrics(bh_res["BuyHoldEquity"], bh_res["DailyReturn"], 252)


# ----------------------------------------------------------------------------
# 🚦 오늘의 신호
# ----------------------------------------------------------------------------
st.subheader("🚦 오늘의 신호")
px = float(signal_df["Close"].iloc[-1])
ma_now, up_now, lo_now = float(ma.iloc[-1]), float(upper.iloc[-1]), float(lower.iloc[-1])
disp = px / ma_now - 1
pos_now = float(position.iloc[-1])
holding = pos_now > 0
asof = signal_df.index[-1].date()

if pos_now >= 1.0:
    st.success(f"### ✅ 풀 보유 (100%) — {trade_name} 추세 양호")
elif pos_now > 0:
    st.warning(f"### ⚠️ 비중 축소 ({pos_now*100:.0f}%) — {trade_name} 경계 (가짜하락 의심 구간)")
else:
    st.error(f"### 💵 현금 (0%) — {trade_name} 추세 이탈/청산")

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{signal_ticker} 현재가", f"{px:,.2f}", f"{ma_kind}{ma_n} 대비 {disp*100:+.1f}%")
c2.metric(f"{ma_kind}{ma_n} (중심선)", f"{ma_now:,.2f}")
c3.metric("🟢 진입선 (위로 가면 풀 보유)", f"{up_now:,.2f}")
c4.metric("🔴 청산선 (연속 이탈 시 현금)", f"{lo_now:,.2f}")

cushion = px / lo_now - 1
st.caption(f"기준일 {asof} · 비중 사다리: **종가≥중심선=100% / 중심선~청산선=50% / 청산선 {confirm_bars}봉 연속이탈 or "
           f"{ma_kind} 꺾임=0%**. 지금 청산선까지 여유 **{cushion*100:+.1f}%** (ATR로 변동성 반영, 3배는 약 3배로 움직임).")

if disp > 0.25 and holding:
    st.warning(f"⚠️ **과열 주의**: 현재가가 {ma_n}일선보다 **+{disp*100:.0f}%** 위입니다. 탈출선이 멀어서 "
               f"조정 시 청산 전에 큰 손실을 볼 수 있어요. **신규 추격매수는 자제**하고, 이미 보유 중이면 "
               f"**비중 축소·분할 익절**을 고려하세요. (눌렸다 다시 올라탈 때 진입하면 손절선이 가까워 안전)")


# ----------------------------------------------------------------------------
# 성과 (전략 vs 보유) + 비중 적용
# ----------------------------------------------------------------------------
st.subheader("📊 성과 — 전략 vs 그냥 보유")
period_txt = f"{trade_df.index[0].date()} ~ {trade_df.index[-1].date()} ({len(trade_df):,}일)"
m1, m2, m3, m4 = st.columns(4)
m1.metric("총수익률", f"{metrics.get('total_return', 0)*100:,.0f}%",
          f"보유 {bh_metrics.get('total_return', 0)*100:,.0f}%")
m2.metric("CAGR(연복리)", f"{metrics.get('cagr', 0)*100:,.1f}%",
          f"보유 {bh_metrics.get('cagr', 0)*100:,.1f}%")
m3.metric("샤프(위험대비)", f"{metrics.get('sharpe', 0):,.2f}",
          f"보유 {bh_metrics.get('sharpe', 0):,.2f}")
m4.metric("MDD(최대낙폭)", f"{metrics.get('max_dd', 0)*100:,.0f}%",
          f"보유 {bh_metrics.get('max_dd', 0)*100:,.0f}%", delta_color="inverse")
st.caption(f"기간: {period_txt} · 신호 {signal_ticker} {ma_n}일선 → 거래 {trade_name}")

mdd_strat = metrics.get("max_dd", 0) * weight
mdd_bh = bh_metrics.get("max_dd", 0) * weight
st.info(f"💼 **비중 {weight*100:.0f}%** 적용 시 내 계좌 기준 최대낙폭 — "
        f"**전략 {mdd_strat*100:,.0f}%** / 그냥 보유 {mdd_bh*100:,.0f}%. "
        f"(나머지 {100-weight*100:.0f}%는 현금/안전자산으로 두는 가정)")


# ----------------------------------------------------------------------------
# 차트
# ----------------------------------------------------------------------------
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                    row_heights=[0.55, 0.45],
                    subplot_titles=(f"{signal_ticker} + {ma_kind}{ma_n}·ATR적응밴드 (비중↑▲/↓▼ 시점)",
                                    f"{trade_name} 자산곡선 — 전략 vs 보유 (로그)"))

# row1: 신호 종목 + MA + ATR 적응밴드 + 매매시점
fig.add_trace(go.Scattergl(x=signal_df.index, y=signal_df["Close"], name=signal_ticker,
                           line=dict(color="#333333", width=1.2)), row=1, col=1)
fig.add_trace(go.Scattergl(x=signal_df.index, y=ma, name=f"{ma_kind}{ma_n}",
                           line=dict(color="#1f77b4", width=1.3)), row=1, col=1)
fig.add_trace(go.Scattergl(x=signal_df.index, y=upper, name="진입선(+ATR)", hoverinfo="skip",
                           line=dict(color="#2ca02c", width=0.8, dash="dot")), row=1, col=1)
fig.add_trace(go.Scattergl(x=signal_df.index, y=lower, name="청산선(−ATR)", hoverinfo="skip",
                           line=dict(color="#d62728", width=0.8, dash="dot")), row=1, col=1)
pos = result["Position"]
entries = result.index[pos.diff() > 0]
exits = result.index[pos.diff() < 0]
ent = entries.intersection(signal_df.index)
exi = exits.intersection(signal_df.index)
fig.add_trace(go.Scattergl(x=ent, y=signal_df.loc[ent, "Close"], mode="markers", name="비중↑(매수)",
                           marker=dict(symbol="triangle-up", color="#2ca02c", size=11,
                                       line=dict(width=0.6, color="white"))), row=1, col=1)
fig.add_trace(go.Scattergl(x=exi, y=signal_df.loc[exi, "Close"], mode="markers", name="비중↓(축소·청산)",
                           marker=dict(symbol="triangle-down", color="#d62728", size=11,
                                       line=dict(width=0.6, color="white"))), row=1, col=1)

# 거래량 폭증(투매) 단기 바닥 후보
if show_vol and signal_df["Volume"].sum() > 0:
    vret = signal_df["Close"].pct_change()
    volma = signal_df["Volume"].rolling(20).mean()
    capit = signal_df.index[(vret < 0) & (signal_df["Volume"] > vol_mult * volma)]
    fig.add_trace(go.Scattergl(x=capit, y=signal_df.loc[capit, "Low"] * 0.97, mode="markers",
                               name="거래량 폭증(투매 바닥?)",
                               marker=dict(symbol="star", color="#ff9800", size=11,
                                           line=dict(width=0.5, color="white"))), row=1, col=1)

# row2: 자산곡선
fig.add_trace(go.Scattergl(x=result.index, y=result["StrategyEquity"], name="전략",
                           line=dict(color="#2ca02c", width=1.7)), row=2, col=1)
fig.add_trace(go.Scattergl(x=result.index, y=result["BuyHoldEquity"], name="그냥 보유",
                           line=dict(color="#999999", width=1.4, dash="dash")), row=2, col=1)

fig.update_layout(height=760, hovermode="x", legend=dict(orientation="h"),
                  margin=dict(t=46, b=10, l=10, r=10), dragmode="pan", uirevision="keep")
fig.update_yaxes(type="log", row=1, col=1)
fig.update_yaxes(type="log", title_text="자산(시작=1)", row=2, col=1)
fig.update_xaxes(rangeselector=dict(buttons=[
    dict(count=1, label="1Y", step="year", stepmode="backward"),
    dict(count=5, label="5Y", step="year", stepmode="backward"),
    dict(step="all", label="전체"),
]), row=1, col=1)
st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
st.caption("🖱️ 휠=확대 · 드래그=이동 · 더블클릭=리셋. ▲ 매수 / ▼ 매도 시점. 가격축은 로그스케일(레버리지 장기 비교).")


# ----------------------------------------------------------------------------
# 규칙 요약
# ----------------------------------------------------------------------------
with st.expander("📖 이 전략 규칙 & 주의 (클릭)"):
    st.markdown(
        f"""
### 매일 1분 체크 (기준: {signal_ticker} 종가, {ma_kind}{ma_n})
1. **종가 ≥ {ma_kind}{ma_n} (중심선 {ma_now:,.2f})** → **{trade_name} 풀 보유(100%)**
2. **중심선 아래 ~ 청산선({lo_now:,.2f}) 사이** → **비중 축소(50%)** — 가짜하락 의심, 절반만 빼고 관망
3. **청산선을 {confirm_bars}봉 연속 이탈, 또는 {ma_kind}{ma_n}가 꺾임** → **전량 현금(0%)** — 진짜 추세붕괴
4. 신규 진입: **종가 > 진입선({up_now:,.2f})** + {ma_kind} 상승 중 일 때만. 청산 후 {cooldown}봉은 재진입 금지.
5. **"이번엔 다르다" 금지** — 규칙만 기계적으로.

### 🔬 가짜하락을 어떻게 더 정밀하게 거르나
- **변동성 적응 밴드**: 청산선 = {ma_kind} − {k_exit:g}×ATR. 고정 %가 아니라 **그날의 변동성(ATR)에 비례** →
  변동성 큰 장에선 밴드가 자동으로 넓어져 '평소 출렁임'에 안 털립니다.
- **지속 확인 {confirm_bars}봉**: 청산선을 하루 깨는 건 무시, **연속 {confirm_bars}봉** 깨져야 진짜로 인정 → 하루짜리 흔들기 필터.
- **{ma_kind} 기울기 게이트**: {ma_kind}가 아직 상승 중이면 그 밑 하락도 '눌림(가짜)'으로 보고 **절반만 축소**,
  {ma_kind}가 꺾였으면 '추세붕괴(진짜)'로 보고 **즉시 전량 청산**.
- 핵심: **레버리지로 추세를 길게 타되, '진짜 폭락'만 정확히 회피** — 가짜에는 절반만 반응해 휩쏘 손실을 줄입니다.

### ⚠️ 주의
- 3배는 정밀 필터를 써도 낙폭이 큽니다 → **비중(위 슬라이더)으로 위험 조절**. 못 견딜 것 같으면 50% 이하.
- 갭하락(하룻밤 폭락)은 청산 전에 당할 수 있어요. EMA·짧은 확인봉으로 더 빨리 빠질 순 있지만 가짜에도 민감해집니다.
- 합성·과거 기준이라 실전(차입비용·슬리피지)은 더 나쁠 수 있습니다. 과거 성과가 미래를 보장하지 않습니다. 투자 권유 아님.
        """
    )

# ----------------------------------------------------------------------------
# 🆚 전략 비교 (내 추세추종 vs 메모의 RSI 전략 vs 보유)
# ----------------------------------------------------------------------------
st.subheader("🆚 전략 비교 — 데이터로 직접 검증")
st.caption("같은 종목·기간·수수료로 **① 내 정밀추세 · ② 단순 RSI30/70 · ③ RSI+200MA 롱only(영상 메모 전략) · "
           "④ 그냥 보유** 를 비교합니다. ‘좋아 보이는 전략’이 정말 보유를 이기는지 직접 확인하세요.")

# 합성(H=L=C)은 ATR·RSI가 degenerate → 실제 OHLC가 있는 신호 종목으로 비교
if trade_code == "__synth__":
    comp_df, comp_name = signal_df, f"{signal_ticker}(합성 대신 실제가)"
else:
    comp_df, comp_name = trade_df, trade_name

if st.checkbox("전략 비교 실행 (계산 잠시)", value=False):
    if len(comp_df) < 250:
        st.warning("비교에는 최소 250봉 이상 데이터가 필요합니다.")
    else:
        cmp_tbl, curves = compare_strategies(comp_df, fee_bps, spec)
        fmt = cmp_tbl.copy()
        for c in ("총수익률", "CAGR", "MDD"):
            fmt[c] = fmt[c].map(lambda v: f"{v*100:,.0f}%" if abs(v) >= 1 else f"{v*100:,.1f}%")
        fmt["샤프"] = fmt["샤프"].map(lambda v: f"{v:,.2f}")
        st.dataframe(fmt, use_container_width=True, hide_index=True)

        cfig = go.Figure()
        colors = {"① 정밀추세 (내 전략)": "#2ca02c", "② 단순 RSI 30/70": "#ff9800",
                  "③ RSI+200MA 롱only (메모)": "#1f77b4", "④ 그냥 보유": "#999999"}
        for name, eq in curves.items():
            cfig.add_trace(go.Scattergl(x=eq.index, y=eq, name=name,
                                        line=dict(color=colors.get(name), width=1.6)))
        cfig.update_yaxes(type="log", title_text="자산(시작=1)")
        cfig.update_layout(height=420, hovermode="x", legend=dict(orientation="h"),
                           margin=dict(t=10, b=10, l=10, r=10),
                           title=f"{comp_name} — 전략별 자산곡선 (로그)")
        st.plotly_chart(cfig, use_container_width=True, config=CHART_CONFIG)
        st.caption(f"대상: {comp_name} · {comp_df.index[0].date()}~{comp_df.index[-1].date()} · "
                   f"수수료 {fee_bps:g}bp. ⚠️ 무료 데이터는 **일봉** 기준 — RSI 평균회귀는 1시간봉에서 더 유리할 수 있어 "
                   f"메모(③) 결과가 과소평가될 수 있습니다. 단, ③의 **낮은 낙폭(MDD)** 경향은 일봉에서도 확인됩니다.")

csv = result.round(4).to_csv().encode("utf-8-sig")
st.download_button("⬇ 백테스트 결과 CSV", csv, file_name=f"{trade_name}_trend.csv", mime="text/csv")
