"""
차트 패턴 탐지 (UI 비의존)
==========================
- 캔들 반전 패턴: 도지·망치형·유성형·상승/하락 장악형·샛별/석별형
- 추세/구조: 골든·데드크로스(50/200), 스윙 고점/저점, 지지/저항선
Streamlit에 의존하지 않아 단독 import/테스트가 가능하다.
"""

import numpy as np
import pandas as pd

from engine import compute_rsi


def detect_candle_patterns(df: pd.DataFrame) -> dict:
    """반환: {패턴명: (불리언 Series, 'bull'|'bear'|'neutral')}."""
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    rng = (h - l).replace(0, np.nan)
    body = c - o
    ab = body.abs()
    upper = h - pd.concat([c, o], axis=1).max(axis=1)
    lower = pd.concat([c, o], axis=1).min(axis=1) - l
    pc, po = c.shift(1), o.shift(1)
    pbody = pc - po
    c2, o2 = c.shift(2), o.shift(2)
    mid2 = (o2 + c2) / 2.0

    doji = ab <= 0.1 * rng
    hammer = (lower >= 2 * ab) & (upper <= ab) & (ab > 0) & ~doji
    star = (upper >= 2 * ab) & (lower <= ab) & (ab > 0) & ~doji          # 유성/역망치
    bull_eng = (body > 0) & (pbody < 0) & (c >= po) & (o <= pc)
    bear_eng = (body < 0) & (pbody > 0) & (o >= pc) & (c <= po)
    small1 = ab.shift(1) <= 0.5 * ab.shift(2)
    morning = (c2 < o2) & small1 & (body > 0) & (c > mid2)              # 샛별형
    evening = (c2 > o2) & small1 & (body < 0) & (c < mid2)              # 석별형

    return {
        "도지": (doji.fillna(False), "neutral"),
        "망치형": (hammer.fillna(False), "bull"),
        "유성형": (star.fillna(False), "bear"),
        "상승장악형": (bull_eng.fillna(False), "bull"),
        "하락장악형": (bear_eng.fillna(False), "bear"),
        "샛별형": (morning.fillna(False), "bull"),
        "석별형": (evening.fillna(False), "bear"),
    }


def detect_crosses(df: pd.DataFrame, short: int = 50, long: int = 200):
    """골든/데드 크로스. 반환: (MA단기, MA장기, 골든 시점 Index, 데드 시점 Index)."""
    ms = df["Close"].rolling(short).mean()
    ml = df["Close"].rolling(long).mean()
    above = (ms > ml).astype(float)
    chg = above.diff()
    golden = df.index[chg == 1.0]
    death = df.index[chg == -1.0]
    return ms, ml, golden, death


def swing_points(df: pd.DataFrame, k: int = 10):
    """좌우 k봉 안에서 최고가/최저가인 봉 = 스윙 고점/저점. 반환: (고점 Index, 저점 Index)."""
    w = 2 * k + 1
    hh = df["High"].rolling(w, center=True).max()
    ll = df["Low"].rolling(w, center=True).min()
    sh = df.index[(df["High"] >= hh) & hh.notna()]
    sl = df.index[(df["Low"] <= ll) & ll.notna()]
    return sh, sl


def swing_k_for(n: int) -> int:
    return int(max(5, min(60, n // 250)))


def ascending_channel(df: pd.DataFrame, k: int | None = None):
    """상승하는 전저점(higher-low) 두 개를 이어 '상승 추세선' + 같은 기울기 평행 채널을 만든다.
    반환: dict(lower=(x0,x1,y0,y1), upper=(...), lows=[(x,y),(x,y)], slope) 또는 None.
    lower(아래 추세선)이 '바닥 매수 관심선', upper(평행선)가 채널 상단."""
    if k is None:
        k = swing_k_for(len(df))
    _, sl = swing_points(df, k)
    sl = list(sl)
    if len(sl) < 2:
        return None
    pos = {ts: p for p, ts in enumerate(df.index)}
    lows = df["Low"]
    # 가장 최근의 '올라가는' 연속 저점 쌍(low_b > low_a)을 찾는다.
    pair = None
    for j in range(len(sl) - 1, 0, -1):
        a, b = sl[j - 1], sl[j]
        if lows[b] > lows[a]:
            pair = (a, b)
            break
    if pair is None:
        return None
    a, b = pair
    p0, p1 = pos[a], pos[b]
    y0, y1 = float(lows[a]), float(lows[b])
    if p1 == p0:
        return None
    slope = (y1 - y0) / (p1 - p0)
    last = len(df.index) - 1
    x_end = df.index[-1]

    def lower_at(p):
        return y0 + slope * (p - p0)

    seg_pos = np.arange(p0, last + 1)
    highs = df["High"].to_numpy()[p0:last + 1]
    offset = float((highs - (y0 + slope * (seg_pos - p0))).max())  # 평행 상단까지 거리
    return {
        "lower": (a, x_end, y0, lower_at(last)),
        "upper": (a, x_end, y0 + offset, lower_at(last) + offset),
        "lows": [(a, y0), (b, y1)],
        "slope": slope,
    }


def rsi_pivot_lows(df: pd.DataFrame, period: int = 14, oversold: float = 30, span: int = 3):
    """RSI가 과매도(oversold) 아래에서 '저점(국소 최소)'을 찍은 봉들 → 그 시점의 가격 저점.
    사용자 노하우: RSI 극저 저점들을 이어 채널을 만든다. 반환: [(timestamp, price_low)]."""
    rsi = compute_rsi(df["Close"], period).to_numpy()
    low = df["Low"].to_numpy()
    n = len(rsi)
    out = []
    for i in range(span, n - span):
        r = rsi[i]
        if np.isnan(r) or r >= oversold:
            continue
        if r == np.nanmin(rsi[i - span:i + span + 1]):   # RSI 국소 최저(방향전환 저점)
            out.append((df.index[i], float(low[i])))
    return out


def rsi_channel(df: pd.DataFrame, period: int = 14, oversold: float = 30, span: int = 3):
    """RSI 저점 두 개를 이어 만든 '예측 채널'. 반환: ascending_channel과 같은 형식 dict 또는 None."""
    piv = rsi_pivot_lows(df, period, oversold, span)
    if len(piv) < 2:
        return None
    (ta, ya), (tb, yb) = piv[-2], piv[-1]
    pos = {ts: p for p, ts in enumerate(df.index)}
    p0, p1 = pos[ta], pos[tb]
    if p1 == p0:
        return None
    slope = (yb - ya) / (p1 - p0)
    last = len(df.index) - 1
    seg_pos = np.arange(p0, last + 1)
    highs = df["High"].to_numpy()[p0:last + 1]
    offset = float((highs - (ya + slope * (seg_pos - p0))).max())

    def low_at(p):
        return ya + slope * (p - p0)

    return {
        "lower": (ta, df.index[-1], ya, low_at(last)),
        "upper": (ta, df.index[-1], ya + offset, low_at(last) + offset),
        "lows": [(ta, ya), (tb, yb)],
        "slope": slope,
    }


def descending_channel(df: pd.DataFrame, k: int | None = None):
    """고점이 점점 낮아지는(lower-high) 두 고점을 이어 '하락 추세선(저항)' + 평행 채널.
    이 저항선을 위로 돌파하면 매수 신호. 반환: dict(upper, lower, highs, slope) 또는 None."""
    if k is None:
        k = swing_k_for(len(df))
    sh, _ = swing_points(df, k)
    sh = list(sh)
    if len(sh) < 2:
        return None
    pos = {ts: p for p, ts in enumerate(df.index)}
    highs = df["High"]
    pair = None
    for j in range(len(sh) - 1, 0, -1):
        a, b = sh[j - 1], sh[j]
        if highs[b] < highs[a]:            # 고점이 낮아지는 쌍
            pair = (a, b)
            break
    if pair is None:
        return None
    a, b = pair
    p0, p1 = pos[a], pos[b]
    y0, y1 = float(highs[a]), float(highs[b])
    if p1 == p0:
        return None
    slope = (y1 - y0) / (p1 - p0)           # 음수(하락)
    last = len(df.index) - 1

    def upper_at(p):
        return y0 + slope * (p - p0)

    seg_pos = np.arange(p0, last + 1)
    lows = df["Low"].to_numpy()[p0:last + 1]
    offset = float(((y0 + slope * (seg_pos - p0)) - lows).max())
    return {
        "upper": (a, df.index[-1], y0, upper_at(last)),      # 하락 저항선
        "lower": (a, df.index[-1], y0 - offset, upper_at(last) - offset),
        "highs": [(a, y0), (b, y1)],
        "slope": slope,
    }


def support_resistance(df: pd.DataFrame, k: int | None = None, n_levels: int = 3):
    """최근 스윙 저점=지지, 스윙 고점=저항 레벨. 반환: (지지 리스트, 저항 리스트)."""
    if k is None:
        k = swing_k_for(len(df))
    sh, sl = swing_points(df, k)
    res = list(df.loc[sh, "High"].tail(n_levels).round(2)) if len(sh) else []
    sup = list(df.loc[sl, "Low"].tail(n_levels).round(2)) if len(sl) else []
    return sup, res
