#!/usr/bin/env python3
"""Diagnose signal conditions against live OKX data using ccxt directly."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import ccxt
import pandas as pd
import talib
import numpy as np

# Load API creds from env
# Load .env manually
if os.path.exists(".env"):
    for line in open(".env"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

exchange = ccxt.okx({
    "apiKey":   os.environ.get("OKX_API_KEY", ""),
    "secret":   os.environ.get("OKX_SECRET", ""),
    "password": os.environ.get("OKX_PASSWORD", ""),
    "enableRateLimit": True,
})

pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "LINK/USDT"]

def fetch(pair, tf, limit):
    raw = exchange.fetch_ohlcv(pair, tf, limit=limit)
    df = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
    return df

def vwap_rolling(df, window=288):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"]
    return (tp * vol).rolling(window).sum() / vol.rolling(window).sum()

def bb_middle(df, window=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return tp.rolling(window).mean()

for pair in pairs:
    try:
        df5  = fetch(pair, "5m",  400)
        df1h = fetch(pair, "1h",  250)

        # 5m indicators
        close_a = df5["close"].values.astype(float)
        high_a  = df5["high"].values.astype(float)
        low_a   = df5["low"].values.astype(float)
        vol_a   = df5["volume"].values.astype(float)

        rsi    = float(talib.RSI(close_a, timeperiod=14)[-1])
        adx    = float(talib.ADX(high_a, low_a, close_a, timeperiod=14)[-1])
        ema20  = float(talib.EMA(close_a, timeperiod=20)[-1])
        ema50  = float(talib.EMA(close_a, timeperiod=50)[-1])
        vol    = float(vol_a[-1])
        vol_avg= float(pd.Series(vol_a).rolling(30).mean().iloc[-1])
        vwap   = float(vwap_rolling(df5, 288).iloc[-1])
        bb_mid = float(bb_middle(df5, 20).iloc[-1])
        close  = float(close_a[-1])

        # 1h indicators
        c1h_a   = df1h["close"].values.astype(float)
        e50_1h  = float(talib.EMA(c1h_a, timeperiod=50)[-1])
        e200_1h = float(talib.EMA(c1h_a, timeperiod=200)[-1])

        print(f"\n{'='*60}")
        print(f"  {pair} @ {close:.4f}")
        print(f"  RSI={rsi:.1f}  ADX={adx:.1f}  Vol/Avg={vol/vol_avg:.2f}x")
        print(f"  EMA20={ema20:.4f}  EMA50_5m={ema50:.4f}")
        print(f"  EMA50_1h={e50_1h:.4f}  EMA200_1h={e200_1h:.4f}")
        print(f"  VWAP={vwap:.4f}  BB_mid={bb_mid:.4f}")

        # Signal 4: momentum_breakout (no volume filter)
        s4 = {
            "close > VWAP":      (close > vwap,      f"{close:.2f} > {vwap:.2f}"),
            "close > EMA50_1h":  (close > e50_1h,    f"{close:.2f} > {e50_1h:.2f}"),
            "close > BB_mid":    (close > bb_mid,     f"{close:.2f} > {bb_mid:.2f}"),
            "55 < RSI < 80":     (55 < rsi < 80,      f"RSI={rsi:.1f}"),
            "ADX > 20":          (adx > 20,           f"ADX={adx:.1f}"),
        }
        s4_pass = all(v for v,_ in s4.values())
        print(f"\n  Signal 4 (momentum_breakout): {'✅ WOULD FIRE' if s4_pass else '❌ BLOCKED'}")
        for k,(v,detail) in s4.items():
            print(f"    {'✅' if v else '❌'} {k}  ({detail})")

        # Signal 5: trend_follow (no volume filter)
        s5 = {
            "EMA20 > EMA50_5m":  (ema20 > ema50,     f"{ema20:.2f} > {ema50:.2f}"),
            "close > EMA20":     (close > ema20,      f"{close:.2f} > {ema20:.2f}"),
            "close > EMA50_1h":  (close > e50_1h,     f"{close:.2f} > {e50_1h:.2f}"),
            "40 < RSI < 75":     (40 < rsi < 75,      f"RSI={rsi:.1f}"),
            "ADX > 20":          (adx > 20,           f"ADX={adx:.1f}"),
        }
        s5_pass = all(v for v,_ in s5.values())
        print(f"\n  Signal 5 (trend_follow): {'✅ WOULD FIRE' if s5_pass else '❌ BLOCKED'}")
        for k,(v,detail) in s5.items():
            print(f"    {'✅' if v else '❌'} {k}  ({detail})")

    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"{pair}: Error — {e}")

print("\nDone.")
