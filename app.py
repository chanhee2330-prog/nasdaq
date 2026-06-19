"""
나스닥/반도체 백테스팅 웹앱
==========================
- 신호 종목(NQ=F, SOXX 등)으로 매매 신호를 만들고
- 실제 손익은 거래 종목(SOXL 등 레버리지 ETF)으로 계산
- 돈치안 돌파 진입 + ATR 트레일링 스톱(수익 길게) + ADX/이격도 필터
- '매수 후 보유'를 이기는 전략 자동 탐색

실행:  streamlit run app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from engine import (
    build_position, compute_metrics, map_position, optimize,
    run_backtest, signal_buy_and_hold, synth_leverage_df, walk_forward,
)

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
    "합성 레버리지 (신호 지수 ×N, 전체기간)": "__synth__",
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
# 사이드바
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ 설정")
mode = st.sidebar.radio(
    "모드", ["📊 단일 전략 백테스트", "🔍 전략 자동 탐색", "🔬 워크포워드 검증"],
    index=0,
    help="단일 전략 = 전략 하나를 자세히 보기 / 자동 탐색 = 여러 전략 순위 / 워크포워드 = 과최적화 배제 검증",
)

st.sidebar.markdown("**종목**")
sig_label = st.sidebar.selectbox("신호 종목 (추세 판단)", list(SIGNAL_TICKERS.keys()), index=1,
                                 help="매수/매도 '신호'를 만드는 기준 종목. 덜 흔들리는 기초자산(지수)을 권장.")
signal_ticker = SIGNAL_TICKERS[sig_label]
if signal_ticker == "__custom__":
    signal_ticker = st.sidebar.text_input("신호 티커", value="NQ=F").strip().upper()

trd_label = st.sidebar.selectbox("거래 종목 (실제 매매)", list(TRADE_TICKERS.keys()), index=1,
                                 help="실제로 사고파는 종목. 손익은 이 종목 가격으로 계산됨(SOXL=반도체 3배).")
trade_ticker = TRADE_TICKERS[trd_label]
synth_mult = None
if trade_ticker == "__synth__":
    synth_mult = st.sidebar.slider("합성 레버리지 배수 (×)", 1.0, 3.0, 3.0, 0.5)
    trade_ticker = f"{signal_ticker}×{synth_mult:g} 합성"
elif trade_ticker == "__same__":
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
        "전략", ["추세추종 (MA) ⭐추천", "돌파+트레일", "이동평균 교차 (SMA)", "RSI", "매수 후 보유"],
        help="여러 전략을 검증한 결과, 단순 '추세추종(장기 이동평균)'이 가장 안정적으로 우수했습니다.")
    if single_strategy == "추세추종 (MA) ⭐추천":
        sp["window"] = st.sidebar.slider("이동평균 기간 (일)", 3, 250, 200,
                                         help="종가가 이 기간의 이동평균선 '위'면 보유, '아래'면 현금. 200 권장.")
        sp["buffer"] = st.sidebar.slider("버퍼 밴드 (%)", 0.0, 10.0, 5.0, 0.5,
                                         help="MA를 이 % 이상 확실히 돌파할 때만 매매. 잦은 매매(휩쏘)를 줄여 "
                                              "수익↑·낙폭↓·거래수↓. 3~5%% 권장.") / 100.0
    elif single_strategy == "돌파+트레일":
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
    elif single_strategy == "이동평균 교차 (SMA)":
        sp["short"] = st.sidebar.slider("단기", 3, 100, 20)
        sp["long"] = st.sidebar.slider("장기", 10, 300, 100)
    elif single_strategy == "RSI":
        sp["period"] = st.sidebar.slider("RSI 기간", 5, 30, 14)
        sp["ma_period"] = st.sidebar.slider("RSI 이동평균", 2, 30, 9)
        sp["oversold"] = st.sidebar.slider("과매도", 10, 45, 30)
        sp["overbought"] = st.sidebar.slider("과매수", 55, 90, 70)
elif mode == "🔬 워크포워드 검증":
    sort_key = st.sidebar.selectbox("최적화 기준(학습 구간)", ["CAGR", "총수익률", "샤프"], index=0)
    train_years = st.sidebar.number_input("학습 기간 (년)", 0.5, 20.0, 3.0, 0.5)
    test_years = st.sidebar.number_input("검증 기간 (년)", 0.25, 5.0, 1.0, 0.25)
else:
    sort_key = st.sidebar.selectbox("순위 기준", ["CAGR", "총수익률", "샤프"], index=0)

run = st.sidebar.button("▶ 실행", type="primary", use_container_width=True)


# ----------------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------------
st.title("📈 나스닥 → SOXL 추세추종 백테스팅")
st.caption("반도체 지수(SOXX) 추세로 신호를 만들어 SOXL(반도체 3배 레버리지)을 매매하는 전략을 "
           "과거 데이터로 검증합니다. 교육·연구용이며 투자 권유가 아닙니다.")

with st.expander("📖 처음이신가요? — 이 앱과 기본값 설명 (클릭해서 펼치기)", expanded=not run):
    st.markdown(
        """
### 이 앱이 하는 일
**백테스트** = "이 매매 규칙을 과거에 그대로 따랐다면 결과가 어땠을까?"를 실제 데이터로 계산해 보는 것입니다.
그냥 사서 들고 있는 것(**매수 후 보유**)과 비교해, 규칙 매매가 더 나은지 확인합니다.

### 🟢 기본값(추천 설정) — 그대로 ▶ 실행만 눌러도 됩니다
| 항목 | 기본값 | 뜻 |
|---|---|---|
| 모드 | **단일 전략 백테스트** | 전략 하나를 자세히 보기 |
| 신호 종목 | **SOXX** (반도체 지수) | 매수/매도 *신호*를 만드는 기준 (덜 흔들리는 지수) |
| 거래 종목 | **SOXL** (반도체 3배) | 실제로 사고파는 종목 (손익 계산 대상) |
| 봉 기준 | **일봉 · 전체기간** | 하루 단위, 데이터가 있는 처음부터 끝까지 |
| 전략 | **추세추종(MA) 200 + 버퍼 5%** ⭐ | 아래 설명 참고 |

### ⭐ 추천 전략: 추세추종 (200일 이동평균 + 버퍼 밴드)
> **규칙**: 신호 종목의 종가가 **200일 이동평균선을 5% 이상 확실히 넘으면 SOXL 보유(매수), 5% 아래로 내려가면 전량 매도(현금)**.
>
> '버퍼 5%'는 이동평균선 근처에서 사고팔고를 반복하는 **휩쏘(whipsaw)를 줄이는 장치**입니다.
> 검증 결과 거래 횟수를 1/5로 줄이면서 **수익은 늘고 낙폭은 줄었습니다**.

왜 이게 추천일까요? 여러 전략(RSI·돌파·채널·숏 등)을 다 검증해 봤지만, **이 단순한 규칙이 가장 꾸준히 좋았습니다.**
- 레버리지 ETF(SOXL 등)는 **큰 폭락 한 번에 -90~-100%로 거의 청산**됩니다(회복 불가).
- 추세추종은 **하락 추세가 시작되면 현금으로 빠져 폭락을 피하므로**, 길게 보면 그냥 들고 있는 것보다 훨씬 유리합니다.
- 단, **강한 급등장에서는 그냥 보유가 더 나을 수 있습니다** (잠깐 쉬는 사이 급등을 놓침). 이건 전략의 약점이 아니라 트레이드오프입니다.

### 📊 결과 숫자 읽는 법
- **총수익률**: 기간 전체 누적 수익 (예: +500% = 6배)
- **CAGR**: 연평균 복리 수익률 (매년 평균 몇 % 불었나)
- **MDD(최대 낙폭)**: 고점 대비 최대 하락폭 — **작을수록(0에 가까울수록) 안전**. 레버리지 보유는 보통 -80~-100%
- **샤프 지수**: 위험 대비 효율 — 높을수록 좋음 (1 이상이면 양호)
- 차트의 ▲초록=매수 시점, ▼빨강=매도 시점. 아래 곡선은 전략(초록) vs 보유(회색 점선) 자산 변화.

### 🧪 더 해보기
- **거래 종목 → "합성 레버리지"**: SOXL이 없던 옛날(닷컴·금융위기)까지 "그때 3배 들고 있었다면?"을 봅니다.
- **모드 → 자동 탐색 / 워크포워드**: 여러 전략 순위 비교 / 과최적화를 걸러낸 '진짜' 성과 검증.

> ⚠️ **주의**: 과거 성과가 미래를 보장하지 않습니다. 레버리지 ETF는 매우 위험합니다. 이 도구는 교육·연구용이며 투자 권유가 아닙니다.
        """
    )

if not run:
    st.info("👈 왼쪽 사이드바에서 **▶ 실행** 버튼을 누르면 시작합니다. "
            "기본값(SOXX 신호 → SOXL · 일봉 전체 · 추세추종 200)이 추천 설정이라 그대로 눌러도 됩니다.")
    st.stop()

spin = st.spinner("데이터 불러오는 중...")
with spin:
    signal_df = load_data(signal_ticker, interval, period)
    if synth_mult is not None:
        trade_df = synth_leverage_df(signal_df, synth_mult)  # 신호 지수의 합성 N배
    else:
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

if synth_mult is not None:
    st.info(f"🧪 **합성 레버리지 모드**: '{signal_ticker}' 지수의 일별수익을 {synth_mult:g}배로 "
            f"시뮬레이션(연 1% 비용 가정)한 가상 종목입니다. 실제 ETF가 없던 과거(닷컴·금융위기)까지 "
            f"레버리지 백테스트가 가능합니다. 실제 ETF와는 오차가 있을 수 있습니다.")

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


# ============================ 워크포워드 검증 ==============================
if mode == "🔬 워크포워드 검증":
    st.subheader(f"🔬 워크포워드 검증 — {hdr}")
    st.caption(f"기간: {period_txt} · 봉: {interval_label}")
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
                   f"({len(folds)}개 중 {wins}개 구간만 보유 초과). "
                   f"보유가 강한 급등장 비중이 큰 구간입니다. 최적화 기준을 '샤프'로 바꾸면 "
                   f"낙폭 대비 효율을 볼 수 있습니다.")

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
    st.caption("※ '선택된 전략'이 구간마다 바뀌면 그만큼 안정적인 단일 전략을 찾기 어렵다는 신호입니다. "
               "특정 전략이 자주 선택되면 그 전략의 신뢰도가 높다고 볼 수 있습니다.")
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
    "추세추종 (MA) ⭐추천": {"type": "추세추종(MA)", **sp},
    "돌파+트레일": {"type": "돌파+트레일", "atr_n": sp.get("atr_n", 14), **sp},
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
