"""전략 정면 비교 (일봉 근사 검증)
- ① 정밀추세 (signal_trend_adaptive)  ← 사용자가 검증한 전략
- ② 단순 RSI 30/70 (추세필터 없음)    ← 메모 주장: 장기 손실
- ③ RSI + 200일선 위 롱only           ← 메모의 '유효 전략'
- ④ 그냥 보유 (buy & hold)            ← 벤치마크
같은 종목·기간·수수료(5bp)로 총수익·CAGR·MDD·샤프·거래수를 비교한다.
※ 무료 데이터 한계로 일봉 기준 근사 — 메모의 1시간봉 결과와 정확히 같진 않음.
"""
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

from engine import (
    signal_trend_adaptive, signal_rsi_channel, signal_buy_and_hold,
    run_backtest, compute_metrics,
)


def _session():
    try:
        from curl_cffi import requests as _creq
        return _creq.Session(impersonate="chrome", verify=False)
    except Exception:
        return None


def load(ticker):
    df = yf.download(ticker, period="max", interval="1d",
                     auto_adjust=True, progress=False, session=_session())
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df


def memo_rsi_trend(df):
    """메모 전략: RSI 과매도/과매수 채널 + 200일선 위에서만 보유(롱only)."""
    rsi_pos = signal_rsi_channel(df, period=14, oversold=30, overbought=70)
    ma200 = df["Close"].rolling(200).mean()
    return (rsi_pos.astype(float) * (df["Close"] > ma200).astype(float))


def evaluate(name, df, pos, fee_bps):
    res = run_backtest(df, pos, fee_bps)
    m = compute_metrics(res["StrategyEquity"], res["StrategyReturn"], 252)
    trades = int((res["Trade"] > 0).sum())
    return {"전략": name, "총수익률": m.get("total_return", 0), "CAGR": m.get("cagr", 0),
            "MDD": m.get("max_dd", 0), "샤프": m.get("sharpe", 0), "거래수": trades}


def run(ticker):
    df = load(ticker)
    if df.empty:
        print(f"[{ticker}] 데이터 없음"); return
    period = f"{df.index[0].date()} ~ {df.index[-1].date()} ({len(df):,}일)"
    rows = [
        evaluate("① 정밀추세(MA150)",
                 df, signal_trend_adaptive(df, 150, False, 14, 0.5, 1.5, 2, 10, 3), 5.0),
        evaluate("② 단순 RSI 30/70",
                 df, signal_rsi_channel(df, 14, 30, 70), 5.0),
        evaluate("③ RSI+200MA 롱only",
                 df, memo_rsi_trend(df), 5.0),
        evaluate("④ 그냥 보유",
                 df, signal_buy_and_hold(df), 0.0),
    ]
    out = pd.DataFrame(rows)
    for c in ["총수익률", "CAGR", "MDD"]:
        out[c] = (out[c] * 100).map(lambda v: f"{v:,.0f}%" if abs(v) >= 100 else f"{v:,.1f}%")
    out["샤프"] = out["샤프"].map(lambda v: f"{v:,.2f}")
    print(f"\n===== {ticker}  ({period}) =====")
    print(out.to_string(index=False))


if __name__ == "__main__":
    for t in ("QQQ", "SOXL"):
        run(t)
