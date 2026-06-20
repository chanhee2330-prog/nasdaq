"""
백테스트 엔진 (UI 비의존)
=========================
지표 · 전략 신호 · 신호→거래 정렬 · 백테스트 · 옵티마이저 · 워크포워드 검증.
Streamlit에 의존하지 않으므로 단독 import/테스트가 가능하다.
"""

import numpy as np
import pandas as pd


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


def synth_leverage_df(base_df: pd.DataFrame, mult: float, annual_cost: float = 0.01) -> pd.DataFrame:
    """기초자산 일별수익을 mult배(일별 리밸런싱)한 '합성 레버리지' 가격 시리즈.
    레버리지 ETF가 없던 과거(닷컴·금융위기 등)까지 N배 백테스트를 가능하게 한다.
    annual_cost: 연 운용보수+차입비용 가정(기본 1%)."""
    ret = base_df["Close"].pct_change().fillna(0.0)
    lev = mult * ret - annual_cost / 252.0
    close = (1 + lev).cumprod()
    out = pd.DataFrame(index=base_df.index)
    out["Open"] = out["High"] = out["Low"] = out["Close"] = close
    out["Volume"] = 0.0
    return out


# ----------------------------------------------------------------------------
# 전략 (신호 종목 기준 포지션 1/0 생성)
# ----------------------------------------------------------------------------
def signal_buy_and_hold(df):
    return pd.Series(1.0, index=df.index)


def signal_sma_crossover(df, short, long):
    return (df["Close"].rolling(short).mean() > df["Close"].rolling(long).mean()).astype(float)


def signal_sma_price(df, window, buffer=0.0):
    """종가 > 이동평균이면 보유.
    buffer>0이면 '버퍼 밴드': MA×(1+buffer) 상향 돌파 시 진입, MA×(1-buffer) 하향 이탈 시 청산.
    MA 근처에서의 잦은 매매(휩쏘)를 줄여 거래수↓·수익↑·낙폭↓ 효과."""
    ma = df["Close"].rolling(window).mean()
    if buffer <= 0:
        return (df["Close"] > ma).astype(float)
    c = df["Close"].to_numpy()
    m = ma.to_numpy()
    pos = np.zeros(len(c))
    s = 0
    for i in range(len(c)):
        if np.isnan(m[i]):
            pos[i] = s
            continue
        if s == 0 and c[i] > m[i] * (1 + buffer):
            s = 1
        elif s == 1 and c[i] < m[i] * (1 - buffer):
            s = 0
        pos[i] = s
    return pd.Series(pos, index=df.index)


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
                          disp_cap=115, disp_n=20, cooldown=0):
    """돈치안 돌파 진입 + ATR 트레일링 스톱 청산 (+ADX/이격도 진입 필터).
    오르는 동안은 계속 보유하고, 고점 대비 ATR×k 만큼 밀릴 때만 청산.
    - use_disp: 이격도(현재가/MA×100)가 disp_cap 초과면 '이미 많이 오른 과열' → 신규 매수 금지.
    - cooldown: 청산 후 이 봉 수 동안 재매수 금지 → 고점 근처 반복 진입(churn) 억제."""
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
    cd = 0
    for i in range(n):
        if not in_pos:
            if cd > 0:                         # 쿨다운: 재진입 대기
                cd -= 1
                continue
            enter = (not np.isnan(up[i])) and c[i] > up[i] and not np.isnan(av[i])
            if enter and use_adx and (np.isnan(adxv[i]) or adxv[i] < adx_min):
                enter = False
            if enter and use_disp and (not np.isnan(dv[i]) and dv[i] > disp_cap):
                enter = False                  # 과열(많이 오른) 구간 매수 금지
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
                cd = cooldown                  # 청산 후 쿨다운 시작
            else:
                pos[i] = 1.0
    return pd.Series(pos, index=df.index)


def signal_swing(df, ma_n=20, band_pct=0.05, trend_n=100, use_trend=True, max_hold=10):
    """스윙(눌림목 반등) 매매.
    - (선택) 상승추세 필터: 종가 > 장기MA(trend_n) 일 때만 진입.
    - 진입: 종가가 중심MA(ma_n)의 하단밴드(MA×(1-band_pct)) 이하로 눌릴 때(과매도 눌림목) 매수.
    - 청산: 종가가 중심MA 회복 또는 상단밴드 도달, 또는 max_hold봉 경과 → 매도.
    짧게 사고 짧게 파는 단기 스윙 → 보유기간이 추세추종보다 훨씬 짧다."""
    close = df["Close"]
    ma = close.rolling(ma_n).mean()
    trend = close.rolling(trend_n).mean()
    c = close.to_numpy()
    m = ma.to_numpy()
    tr = trend.to_numpy()
    n = len(c)
    pos = np.zeros(n)
    in_pos, held = False, 0
    for i in range(n):
        if np.isnan(m[i]):
            pos[i] = 0.0
            continue
        lo = m[i] * (1 - band_pct)
        up = m[i] * (1 + band_pct)
        if not in_pos:
            up_ok = (not use_trend) or (not np.isnan(tr[i]) and c[i] > tr[i])
            if up_ok and c[i] <= lo:
                in_pos, held = True, 0
                pos[i] = 1.0
        else:
            held += 1
            if c[i] >= m[i] or c[i] >= up or held >= max_hold:
                in_pos = False
                pos[i] = 0.0
            else:
                pos[i] = 1.0
    return pd.Series(pos, index=df.index)


def signal_trendline_breakout(df, k=10, atr_n=14, atr_k=3.0):
    """하락 추세선(낮아지는 고점들을 이은 저항선) 상향 돌파 매수 + ATR 트레일링 청산.
    - 좌우 k봉 안에서 최고가인 '스윙 고점'을 인과적으로 확정(미래 정보 X).
    - 최근의 '낮아지는 고점' 두 개로 하락 저항선을 긋고, 종가가 그 위로 뚫으면 매수.
    - 청산: 고점 대비 ATR×k 트레일링 스톱."""
    high = df["High"].to_numpy()
    close = df["Close"].to_numpy()
    av = atr(df, atr_n).to_numpy()
    n = len(close)
    pos = np.zeros(n)
    pivots = []          # 확정된 스윙 고점 [(idx, high)]
    in_pos = False
    peak = 0.0
    prev_res = None
    for i in range(n):
        j = i - k                                   # k봉 전 후보
        if j - k >= 0 and high[j] == high[j - k:j + k + 1].max():
            if not pivots or pivots[-1][0] != j:
                pivots.append((j, high[j]))
        # 최근 '낮아지는 고점' 두 개로 저항선 값 계산
        res = None
        for q in range(len(pivots) - 1, 0, -1):
            a0, h0 = pivots[q - 1]
            a1, h1 = pivots[q]
            if h1 < h0:
                res = h1 + (h1 - h0) / (a1 - a0) * (i - a1)
                break
        if not in_pos:
            if (res is not None and prev_res is not None and not np.isnan(av[i])
                    and close[i] > res and close[i - 1] <= prev_res):     # 저항선 상향 돌파
                in_pos = True
                peak = close[i]
                pos[i] = 1.0
        else:
            peak = max(peak, close[i])
            stop = peak - atr_k * av[i] if not np.isnan(av[i]) else -np.inf
            if close[i] < stop:
                in_pos = False
                pos[i] = 0.0
            else:
                pos[i] = 1.0
        prev_res = res
    return pd.Series(pos, index=df.index)


def signal_rsi_channel(df, period=14, oversold=30, overbought=70, ob_persist=0):
    """RSI 채널 매매 (사용자 노하우).
    - 매수: RSI가 '극히 낮음(과매도)'에서 위로 방향을 바꿀 때(저점 확인) → 채널 바닥 매수.
    - 매도: RSI가 '높음(과매수)'에서 아래로 방향을 바꿀 때 → 매도.
      단, ob_persist>0 이면 'RSI가 과매수에 최소 그만큼(봉) 머문 뒤' 꺾일 때만 매도.
      → 짧게 튄 과매수 스파이크는 무시하고(강한 추세는 계속 보유), 오래 높던 RSI가
        꺾일 때(천장 형성)만 판다. '얼마나 오래 높았나'를 반영.
    RSI 저점/고점의 방향전환을 신호로 쓴다(미래 정보 사용 안 함)."""
    rsi = compute_rsi(df["Close"], period).to_numpy()
    n = len(rsi)
    pos = np.zeros(n)
    in_pos = False
    hi = 0  # 연속으로 과매수(>=overbought)에 머문 봉 수
    for i in range(1, n):
        r, rp = rsi[i], rsi[i - 1]
        if np.isnan(r) or np.isnan(rp):
            pos[i] = 1.0 if in_pos else 0.0
            continue
        crossed_down = rp >= overbought > r        # 과매수에서 아래로 꺾임
        if not in_pos and rp <= oversold < r:      # 과매도 탈출(저점 방향전환)
            in_pos = True
        elif in_pos and crossed_down and hi >= ob_persist:  # 충분히 오래 높았다가 꺾임
            in_pos = False
        hi = hi + 1 if r >= overbought else 0       # 결정 후 갱신(꺾인 봉은 0으로 리셋)
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=df.index)


def signal_crash_recovery(df, ma_n=50, dip=0.15, pop=0.15, use_trend=False,
                          trend_n=200, max_hold=0):
    """폭락 매수 + 급등 빠른 청산 (비대칭 역추세).
    - 진입: 종가가 중심MA(ma_n)보다 dip(예 15%) 이상 아래로 '폭락'하면 매수.
    - 보유: 회복하는 동안 길게 보유(인내) — 중간 손절 없음.
    - 청산: 종가가 중심MA보다 pop(예 15%) 이상 위로 '급등(과열)'하면 즉시 청산(빠른 익절).
    - use_trend: 장기MA 위(상승추세)에서의 폭락만 매수(추세 안에서의 눌림목만).
    - max_hold>0: 안전장치로 최대 보유 봉 제한(0=무제한).
    V자 반등 종목(레버리지 ETF 등)에 적합. 단, '길게 보유'는 추가 하락 위험을 감수한다."""
    close = df["Close"]
    ma = close.rolling(ma_n).mean()
    trend = close.rolling(trend_n).mean()
    c = close.to_numpy()
    m = ma.to_numpy()
    tr = trend.to_numpy()
    n = len(c)
    pos = np.zeros(n)
    in_pos, held = False, 0
    for i in range(n):
        if np.isnan(m[i]):
            pos[i] = 0.0
            continue
        if not in_pos:
            ok = (not use_trend) or (not np.isnan(tr[i]) and c[i] > tr[i])
            if ok and c[i] <= m[i] * (1 - dip):       # 폭락 → 매수
                in_pos, held = True, 0
                pos[i] = 1.0
        else:
            held += 1
            if c[i] >= m[i] * (1 + pop) or (max_hold and held >= max_hold):  # 급등 → 빠른 청산
                in_pos = False
                pos[i] = 0.0
            else:
                pos[i] = 1.0                            # 회복까지 길게 보유
    return pd.Series(pos, index=df.index)


def _ma(close: pd.Series, n: int, ema: bool = False) -> pd.Series:
    """SMA 또는 EMA. EMA는 최근 가격에 민감 → 폭락을 몇 봉 더 일찍 감지."""
    return close.ewm(span=n, min_periods=n).mean() if ema else close.rolling(n).mean()


def trend_adaptive_lines(df, window, ema=False, atr_n=14, k_enter=0.5, k_exit=1.5):
    """정밀 추세추종의 진입선/청산선(변동성 적응형).
    - 진입선 = MA + k_enter×ATR  (이 위로 강하게 올라설 때만 신규 진입)
    - 청산선 = MA − k_exit×ATR    (변동성↑ → 밴드 자동 확대 → 평소 출렁임에 안 털림)
    app(차트)·신호 로직이 같은 선을 쓰도록 한곳에서 계산한다."""
    ma = _ma(df["Close"], window, ema)
    a = atr(df, atr_n)
    return ma, ma + k_enter * a, ma - k_exit * a


def signal_trend_adaptive(df, window=150, ema=False, atr_n=14, k_enter=0.5, k_exit=1.5,
                          confirm_bars=2, slope_n=10, cooldown=3):
    """정밀 추세추종 — '진짜 폭락 vs 가짜하락'을 3가지 근거로 판별해 단계적(1.0/0.5/0.0) 대응.
    레버리지로 추세를 길게 타되, 큰 폭락만 정확히 회피하는 게 목표.

    가짜하락 필터:
      1) 변동성 적응 밴드: 청산선=MA−k_exit×ATR → 변동성 큰 장에선 밴드가 넓어져 평소 출렁임에 안 털림.
      2) 지속 확인(confirm_bars): 청산선을 'N봉 연속' 깨야 진짜로 인정 → 하루짜리 가짜하락 무시.
      3) MA 기울기 게이트: MA가 아직 상승 중이면 그 밑 하락은 '눌림(가짜)' → 절반만 축소(0.5),
         MA가 꺾였으면 '추세붕괴(진짜)' → 전량 청산(0.0)으로 즉시 대응.

    비중 사다리:
      - 1.0(풀): 종가 ≥ MA (건강한 추세)
      - 0.5(축소): MA 아래~청산선 사이, 또는 청산선 하향이나 아직 '가짜' 의심 단계
      - 0.0(현금): 청산선을 confirm_bars 연속 깨거나(지속) MA가 꺾였을 때(진짜 붕괴)
    진입: 종가 > 진입선 + MA 상승 중. 청산 뒤 cooldown봉은 재진입 금지(churn 억제)."""
    ma, entry, exitl = trend_adaptive_lines(df, window, ema, atr_n, k_enter, k_exit)
    c = df["Close"].to_numpy()
    m = ma.to_numpy()
    el = entry.to_numpy()
    xl = exitl.to_numpy()
    n = len(c)
    pos = np.zeros(n)
    cur = 0.0
    below = 0   # 청산선 아래 연속 봉 수
    cd = 0      # 재진입 쿨다운
    for i in range(n):
        if np.isnan(m[i]) or np.isnan(el[i]) or np.isnan(xl[i]):
            pos[i] = cur
            continue
        slope_up = i >= slope_n and not np.isnan(m[i - slope_n]) and m[i] > m[i - slope_n]
        if cur <= 0:                                   # 현금 → 진입 판단
            if cd > 0:
                cd -= 1
                pos[i] = 0.0
                continue
            if c[i] > el[i] and slope_up:              # 진입선 위 + 추세 상승
                cur = 1.0
                below = 0
        else:                                          # 보유(1.0/0.5) → 청산/축소 판단
            if c[i] < xl[i]:                           # 청산선 하향
                below += 1
                if below >= confirm_bars or not slope_up:   # 지속됨 or MA 꺾임 → 진짜
                    cur = 0.0
                    cd = cooldown
                    below = 0
                else:                                  # 아직 가짜 의심 → 절반만 축소
                    cur = 0.5
            else:                                      # 청산선 위로 회복
                below = 0
                cur = 1.0 if c[i] >= m[i] else 0.5     # MA 위=풀 / MA 아래=경계(절반)
        pos[i] = cur
    return pd.Series(pos, index=df.index)


def build_position(df, spec):
    t = spec["type"]
    if t == "정밀추세":
        return signal_trend_adaptive(
            df, spec.get("window", 150), spec.get("ema", False), spec.get("atr_n", 14),
            spec.get("k_enter", 0.5), spec.get("k_exit", 1.5),
            spec.get("confirm_bars", 2), spec.get("slope_n", 10), spec.get("cooldown", 3))
    if t == "하락추세선돌파":
        return signal_trendline_breakout(df, spec.get("k", 10), spec.get("atr_n", 14),
                                        spec.get("atr_k", 3.0))
    if t == "RSI채널":
        return signal_rsi_channel(df, spec.get("period", 14), spec.get("oversold", 30),
                                 spec.get("overbought", 70), spec.get("ob_persist", 0))
    if t == "폭락매수+급등청산":
        return signal_crash_recovery(df, spec.get("ma_n", 50), spec.get("dip", 0.15),
                                     spec.get("pop", 0.15), spec.get("use_trend", False),
                                     spec.get("trend_n", 200), spec.get("max_hold", 0))
    if t == "스윙(밴드)":
        return signal_swing(df, spec.get("ma_n", 20), spec.get("band_pct", 0.05),
                            spec.get("trend_n", 100), spec.get("use_trend", True),
                            spec.get("max_hold", 10))
    if t == "SMA교차":
        return signal_sma_crossover(df, spec["short"], spec["long"])
    if t == "추세추종(MA)":
        return signal_sma_price(df, spec["window"], spec.get("buffer", 0.0))
    if t == "RSI":
        return signal_rsi(df, spec["period"], spec["ma_period"], spec["oversold"], spec["overbought"])
    if t == "돌파+트레일":
        return signal_breakout_trail(
            df, spec["donchian_n"], spec["atr_n"], spec["atr_k"],
            spec.get("use_adx", False), spec.get("adx_min", 20),
            spec.get("use_disp", False), spec.get("disp_cap", 115), spec.get("disp_n", 20),
            spec.get("cooldown", 0))
    return signal_buy_and_hold(df)


def long_short_position(pos_long, max_short_bars, short_size=1.0):
    """롱(1)/현금(0) 포지션을 롱/숏 포지션으로 변환.
    매도 신호로 현금이 되는 구간의 '앞부분 max_short_bars 봉'만 숏(-short_size)으로 잡고,
    그 뒤로는 현금(0). → 숏은 롱보다 항상 '짧게'만 보유한다.
    백테스트 수익은 run_backtest가 음수 포지션을 그대로 처리(가격 하락 시 이익)."""
    p = pos_long.to_numpy()
    out = p.astype(float).copy()
    n, i = len(p), 0
    while i < n:
        if p[i] <= 0:                         # 현금(또는 비롱) 구간
            j = i
            while j < n and p[j] <= 0:
                j += 1
            end = min(i + int(max_short_bars), j)
            out[i:end] = -float(short_size)   # 앞부분만 숏
            out[end:j] = 0.0                  # 나머지는 현금
            i = j
        else:
            i += 1
    return pd.Series(out, index=pos_long.index)


def spec_label(spec):
    t = spec["type"]
    if t == "정밀추세":
        ma_kind = "EMA" if spec.get("ema") else "MA"
        return (f"정밀추세({ma_kind}{spec.get('window', 150)}, 청산−{spec.get('k_exit', 1.5):g}ATR, "
                f"확인{spec.get('confirm_bars', 2)}봉)")
    if t == "하락추세선돌파":
        return f"하락추세선돌파(스윙{spec.get('k', 10)}, ATR{spec.get('atr_n', 14)}×{spec.get('atr_k', 3.0)})"
    if t == "RSI채널":
        p = f", 과매수{spec['ob_persist']}봉↑" if spec.get("ob_persist", 0) else ""
        return f"RSI채널({spec.get('period', 14)}, 과매도{spec.get('oversold', 30)}/과매수{spec.get('overbought', 70)}{p})"
    if t == "폭락매수+급등청산":
        tf = ", 추세필터" if spec.get("use_trend") else ""
        return (f"폭락매수(MA{spec.get('ma_n', 50)}, 폭락-{spec.get('dip', 0.15) * 100:g}%/"
                f"급등+{spec.get('pop', 0.15) * 100:g}%{tf})")
    if t == "스윙(밴드)":
        flt = ", 추세필터" if spec.get("use_trend", True) else ""
        return (f"스윙(MA{spec.get('ma_n', 20)}±{spec.get('band_pct', 0.05) * 100:g}%, "
                f"최대{spec.get('max_hold', 10)}봉{flt})")
    if t == "SMA교차":
        return f"SMA교차(단기{spec['short']}/장기{spec['long']})"
    if t == "추세추종(MA)":
        b = spec.get("buffer", 0.0)
        return f"추세추종(MA{spec['window']}" + (f", 버퍼{b*100:g}%)" if b > 0 else ")")
    if t == "RSI":
        return f"RSI({spec['period']},MA{spec['ma_period']},{spec['oversold']}/{spec['overbought']})"
    if t == "돌파+트레일":
        flt = []
        if spec.get("use_adx"):
            flt.append(f"ADX≥{spec['adx_min']}")
        if spec.get("use_disp"):
            flt.append(f"이격도≤{spec['disp_cap']}")
        if spec.get("cooldown", 0):
            flt.append(f"쿨다운{spec['cooldown']}")
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
# 옵티마이저 / 워크포워드
# ----------------------------------------------------------------------------
def strategy_grid():
    specs = []
    for dn in (20, 40, 55):
        for k in (2.0, 3.0):
            specs.append({"type": "돌파+트레일", "donchian_n": dn, "atr_n": 14, "atr_k": k})
            specs.append({"type": "돌파+트레일", "donchian_n": dn, "atr_n": 14, "atr_k": k,
                          "use_adx": True, "adx_min": 20})
    for w in (100, 150, 200, 250):
        specs.append({"type": "추세추종(MA)", "window": w})
    for w in (150, 200):                 # 정밀추세(변동성적응+지속확인+기울기) 변형
        for ke in (1.5, 2.5):
            for ema in (False, True):
                specs.append({"type": "정밀추세", "window": w, "ema": ema, "atr_n": 14,
                              "k_enter": 0.5, "k_exit": ke, "confirm_bars": 2,
                              "slope_n": 10, "cooldown": 3})
    for w in (150, 200, 250):           # 버퍼 밴드 버전(휩쏘 감소)
        for b in (0.03, 0.05):
            specs.append({"type": "추세추종(MA)", "window": w, "buffer": b})
    for s in (10, 20, 30):
        for l in (50, 100, 200):
            specs.append({"type": "SMA교차", "short": s, "long": l})
    for (p, m, ov, ob) in [(14, 9, 30, 70), (14, 5, 35, 65)]:
        specs.append({"type": "RSI", "period": p, "ma_period": m, "oversold": ov, "overbought": ob})
    for ma_n in (10, 20):                      # 스윙(눌림목) 변형들
        for band in (0.04, 0.07):
            for mh in (5, 10):
                specs.append({"type": "스윙(밴드)", "ma_n": ma_n, "band_pct": band,
                              "trend_n": 100, "use_trend": True, "max_hold": mh})
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


def walk_forward(signal_df, trade_df, ppy, fee_bps, sort_key, train_bars, test_bars, progress=None):
    """과거 train_bars로 최적 전략 선택 → 이후 test_bars(처음 보는 구간)에 적용, 반복.
    검증(OOS) 구간 수익만 이어붙여 반환."""
    dates = trade_df.index
    n = len(dates)
    ret_full = trade_df["Close"].pct_change().fillna(0.0)
    oos_returns, fold_rows = [], []

    total_folds = max(1, (n - train_bars + test_bars - 1) // test_bars)
    i, done = train_bars, 0
    while i < n:
        train_start, train_end = dates[i - train_bars], dates[i - 1]
        test_start, test_end = dates[i], dates[min(i + test_bars, n) - 1]
        sig_is = signal_df[(signal_df.index >= train_start) & (signal_df.index <= train_end)]
        trd_is = trade_df.loc[train_start:train_end]
        if len(trd_is) < 30 or len(sig_is) < 30:
            break

        tbl = optimize(sig_is, trd_is, ppy, fee_bps, sort_key)  # 학습 구간에서만 최적화
        if not tbl.empty:
            best = tbl.iloc[0]
            spec = best["_spec"]
            # 검증 구간 포지션: test_end까지의 데이터로 지표 워밍업 후 OOS만 절취
            sig_upto = signal_df[signal_df.index <= test_end]
            pos = map_position(build_position(sig_upto, spec), trade_df.loc[:test_end].index)
            pos_exec = pos.reindex(trade_df.index).fillna(0.0).shift(1).fillna(0.0)
            trades = pos_exec.diff().abs().fillna(0.0)
            sret = ret_full * pos_exec - trades * (fee_bps / 10_000.0)
            oos = sret.loc[test_start:test_end]
            oos_returns.append(oos)
            strat_tr = (1 + oos).prod() - 1
            bh_tr = (1 + ret_full.loc[test_start:test_end]).prod() - 1
            fold_rows.append({
                "검증구간": f"{test_start.date()}~{test_end.date()}",
                "학습구간": f"{train_start.date()}~{train_end.date()}",
                "선택된 전략": best["파라미터"],
                "검증수익률": strat_tr, "보유수익률": bh_tr, "초과": strat_tr - bh_tr,
            })
        done += 1
        if progress is not None:
            progress.progress(min(1.0, done / total_folds))
        i += test_bars

    if not oos_returns:
        return None, None
    wf_ret = pd.concat(oos_returns)
    wf_ret = wf_ret[~wf_ret.index.duplicated(keep="first")].sort_index()
    return wf_ret, pd.DataFrame(fold_rows)
