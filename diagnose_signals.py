#!/usr/bin/env python3
"""Quick diagnostic: show indicator values for top pairs to find what blocks signals."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from freqtrade.configuration import Configuration
from freqtrade.resolvers import ExchangeResolver
from freqtrade.data.dataprovider import DataProvider
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

# Load config
config = Configuration.from_files(["user_data/config.json", "user_data/config_spot.json"])
config["exchange"]["sandbox"] = False

exchange = ExchangeResolver.load_exchange(config)

pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "LINK/USDT"]

for pair in pairs:
    try:
        # Fetch 5m candles
        df = exchange.refresh_latest_ohlcv([(pair, "5m", "spot")], cache=False)
        key = (pair, "5m", "spot")
        if key not in df or df[key].empty:
            print(f"{pair}: No 5m data")
            continue
        ohlcv = df[key].copy()

        # Fetch 1h candles
        df_1h = exchange.refresh_latest_ohlcv([(pair, "1h", "spot")], cache=False)
        key_1h = (pair, "1h", "spot")

        # Calculate 5m indicators
        ohlcv["rsi"] = ta.RSI(ohlcv, timeperiod=14)
        ohlcv["adx"] = ta.ADX(ohlcv)
        ohlcv["ema_20"] = ta.EMA(ohlcv, timeperiod=20)
        ohlcv["ema_50_5m"] = ta.EMA(ohlcv, timeperiod=50)
        ohlcv["volume_mean"] = ohlcv["volume"].rolling(window=30).mean()
        bb = qtpylib.bollinger_bands(qtpylib.typical_price(ohlcv), window=20, stds=2)
        ohlcv["bb_mid"] = bb["mid"]
        ohlcv["bb_upper"] = bb["upper"]
        ohlcv["bb_lower"] = bb["lower"]
        ohlcv["vwap"] = qtpylib.rolling_vwap(ohlcv, window=288)

        # 1h indicators
        inf = df_1h[key_1h].copy() if key_1h in df_1h else None
        ema_50_1h = 0
        ema_200_1h = 0
        btc_safe = "?"
        if inf is not None and not inf.empty:
            inf["ema_50"] = ta.EMA(inf, timeperiod=50)
            inf["ema_200"] = ta.EMA(inf, timeperiod=200)
            ema_50_1h = inf["ema_50"].iloc[-1]
            ema_200_1h = inf["ema_200"].iloc[-1]

        c = ohlcv.iloc[-1]
        close = c["close"]
        rsi = c["rsi"]
        adx = c["adx"]
        ema20 = c["ema_20"]
        ema50 = c["ema_50_5m"]
        vol = c["volume"]
        vol_mean = c["volume_mean"]
        vwap = c["vwap"]
        bb_mid = c["bb_mid"]

        print(f"\n{'='*60}")
        print(f"  {pair} @ {close:.4f}")
        print(f"{'='*60}")
        print(f"  RSI:        {rsi:.1f}    (need 40-70 for S5, 55-75 for S4)")
        print(f"  ADX:        {adx:.1f}    (need > 20)")
        print(f"  EMA20:      {ema20:.4f}  close > ema20? {'✅' if close > ema20 else '❌'}")
        print(f"  EMA50_5m:   {ema50:.4f}  ema20 > ema50? {'✅' if ema20 > ema50 else '❌'}")
        print(f"  EMA50_1h:   {ema_50_1h:.4f}  close > ema50_1h? {'✅' if close > ema_50_1h else '❌'}")
        print(f"  EMA200_1h:  {ema_200_1h:.4f}  ema50_1h > ema200_1h? {'✅' if ema_50_1h > ema_200_1h else '❌'}")
        print(f"  VWAP:       {vwap:.4f}  close > vwap? {'✅' if close > vwap else '❌'}")
        print(f"  BB_mid:     {bb_mid:.4f}  close > bb_mid? {'✅' if close > bb_mid else '❌'}")
        print(f"  Vol/Avg:    {vol/vol_mean:.2f}x  (need > 1.0)")
        print()

        # Check Signal 5 (trend_follow)
        s5_checks = {
            "ema20 > ema50_5m": ema20 > ema50,
            "close > ema20": close > ema20,
            "close > ema50_1h": close > ema_50_1h,
            "RSI 40-70": 40 < rsi < 70,
            "ADX > 20": adx > 20,
            "vol > vol_mean": vol > vol_mean,
        }
        print("  Signal 5 (trend_follow):")
        all_pass = True
        for name, passed in s5_checks.items():
            print(f"    {'✅' if passed else '❌'} {name}")
            if not passed:
                all_pass = False
        print(f"  → {'WOULD FIRE ✅' if all_pass else 'BLOCKED ❌'}")

        # Check Signal 4 (momentum_breakout)
        s4_checks = {
            "close > vwap": close > vwap,
            "close > ema50_1h": close > ema_50_1h,
            "close > bb_mid": close > bb_mid,
            "RSI 55-75": 55 < rsi < 75,
            "ADX > 20": adx > 20,
            "vol > vol_mean": vol > vol_mean,
        }
        print(f"\n  Signal 4 (momentum_breakout):")
        all_pass = True
        for name, passed in s4_checks.items():
            print(f"    {'✅' if passed else '❌'} {name}")
            if not passed:
                all_pass = False
        print(f"  → {'WOULD FIRE ✅' if all_pass else 'BLOCKED ❌'}")

    except Exception as e:
        print(f"{pair}: Error - {e}")

exchange.close()
