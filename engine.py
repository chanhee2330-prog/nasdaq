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
# 옵티마이저 / 워크포워드
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
