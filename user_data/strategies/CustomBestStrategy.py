import numpy as np
import pandas as pd
from pandas import DataFrame

# Freqtrade imports
from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    merge_informative_pair,
    stoploss_from_absolute,
)
from freqtrade.persistence import Trade, Order, PairLocks
from datetime import datetime, timezone, timedelta
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import os
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class TradeDecision(BaseModel):
    reasoning: str = Field(description="A concise 1-sentence technical reason for your decision.")
    decision: bool = Field(
        description="Set to True if the trade setup is highly probable / safe. Set to False if this is likely a trap, dump, or poor setup."  # noqa: E501
    )
    confidence: int = Field(
        description="Confidence score from 0 to 100 for this decision. Above 65 is required for entry."  # noqa: E501
    )


class ExitDecision(BaseModel):
    reasoning: str = Field(
        description="A concise 1-sentence analytical reason why we should hold or sell right now."
    )
    hold_trade: bool = Field(
        description="Set to True to REJECT the exit order and HOLD for more profit because a massive breakout is occurring. Set to False to ALLOW selling and take the money now."  # noqa: E501
    )


class CustomBestStrategy(IStrategy):
    """
    CustomBestStrategy: Designed using context7 optimal templates and best practices.
    Combines RSI (Momentum), Bollinger Bands (Volatility) and TEMA (Trend Tracking)
    """

    # Optimal timeframe for the strategy
    timeframe = "5m"

    # Default: allow shorts. Will be overridden to False automatically in spot mode.
    can_short = True

    def __init__(self, config: dict, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        # Auto-disable shorting when running in spot market
        # This allows the SAME strategy file to work for both the spot and futures bots
        if config.get("trading_mode", "spot") != "futures":
            self.can_short = False
        
        self.last_global_lock_msg = ""
        self.signal_performance_file = "user_data/signal_performance.json"
        self.signal_performance = {}
        self._load_signal_performance()

        # Configure Google Generative AI for trade confirmation
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            self.llm_client = genai.Client(api_key=api_key)
            self.llm_enabled = True
            # Cache: {(pair, side, entry_tag) -> (decision, timestamp)}
            self.ai_candle_cache: dict = {}
            # Daily budget: track calls to cap max spend
            self.ai_daily_budget = int(os.environ.get("AI_DAILY_BUDGET", 50))
            self.usage_file = "user_data/gemini_usage.json"
            self._load_gemini_usage()
            
            logger.info(
                f"📱 Google LLM (Gemini) enabled. Daily budget: {self.ai_daily_budget} calls."
            )
        else:
            self.llm_enabled = False
            logger.warning("⚠️ GOOGLE_API_KEY not found in environment. LLM integration disabled.")



    # Minimal ROI — aligned with hyperopt-optimized values
    # Let winners run initially (20.9%), then gradually accept smaller profits
    minimal_roi = {"0": 0.209, "30": 0.061, "46": 0.015, "145": 0.0}

    # Stoploss — absolute floor (ATR-based custom_stoploss is the real guard)
    # Hyperopt found -0.13 is optimal; -0.05 was too tight, hit by normal 5m noise
    stoploss = -0.13

    # Trailing stoploss — lock in profit at 1% once we're 3% up (giving more room to trend)
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    # Enable ATR-based custom stoploss
    use_custom_stoploss = True

    # Enable DCA (Dollar Cost Averaging) - automatically buy more on dips
    position_adjustment_enable = True

    # Process indicators only for new candles to save compute
    process_only_new_candles = True

    # Startup candle count — needs 288 for VWAP (24h rolling window at 5m)
    startup_candle_count: int = 288

    def _load_gemini_usage(self):
        """Load Gemini API usage from disk to persist across restarts."""
        self.ai_daily_calls = 0
        self.ai_budget_date = str(datetime.now(timezone.utc).date())
        try:
            if os.path.exists(self.usage_file):
                with open(self.usage_file, 'r') as f:
                    data = json.load(f)
                    if data.get('date') == self.ai_budget_date:
                        self.ai_daily_calls = data.get('calls', 0)
        except Exception as e:
            logger.warning(f"⚠️ Could not load Gemini usage: {e}")

    def _save_gemini_usage(self):
        """Save Gemini API usage to disk."""
        try:
            with open(self.usage_file, 'w') as f:
                json.dump({
                    'date': self.ai_budget_date,
                    'calls': self.ai_daily_calls
                }, f)
        except Exception as e:
            logger.error(f"⚠️ Could not save Gemini usage: {e}")

    def _load_signal_performance(self):
        """Load signal performance tracking from disk."""
        try:
            if os.path.exists(self.signal_performance_file):
                with open(self.signal_performance_file, 'r') as f:
                    self.signal_performance = json.load(f)
        except Exception as e:
            logger.warning(f"⚠️ Could not load signal performance: {e}")

    def _save_signal_performance(self):
        """Save signal performance tracking to disk."""
        try:
            with open(self.signal_performance_file, 'w') as f:
                json.dump(self.signal_performance, f, indent=2)
        except Exception as e:
            logger.error(f"⚠️ Could not save signal performance: {e}")

    def record_trade_result(self, enter_tag: str, profit_ratio: float):
        """Record trade result by signal tag for self-improvement."""
        if enter_tag not in self.signal_performance:
            self.signal_performance[enter_tag] = {"wins": 0, "losses": 0, "total_profit": 0.0}
        
        self.signal_performance[enter_tag]["total_profit"] += profit_ratio
        if profit_ratio > 0:
            self.signal_performance[enter_tag]["wins"] += 1
        else:
            self.signal_performance[enter_tag]["losses"] += 1
        
        self._save_signal_performance()

    @property
    def protections(self):
        """
        Advanced Layered Protections for Active Trading:
        Isolates "toxic" performing coins without shutting down the entire bot.
        """
        return [
            {
                # LAYER 1: The "Breather"
                # Wait 10 candles (50 mins) before trying to trade a coin we just exited.
                "method": "CooldownPeriod",
                "stop_duration_candles": 10,
            },
            {
                # LAYER 2: The "Toxic Pair" filter
                # If a specific coin hits its stoploss 2 times in the last 4 hours...
                "method": "StoplossGuard",
                "lookback_period_candles": 48,  # 4 hours
                "trade_limit": 2,
                "stop_duration_candles": 48,  # ...ban THAT specific coin for 4 hours.
                "only_per_pair": True,
            },
            {
                # LAYER 3: The "Loser" filter
                # If a pair has lost trades recently...
                "method": "LowProfitPairs",
                "lookback_period_candles": 144,  # 12 hours
                "trade_limit": 2,  # Reduced from 3 to 2
                "stop_duration_candles": 72,  # Block for 6 hours
                "required_profit": 0.005,  # Lowered threshold
            },
            {
                # LAYER 4: The "Flash Crash" global panic switch
                # If the bot's total wallet drops by 15% in a 24 hour period...
                "method": "MaxDrawdown",
                "lookback_period_candles": 288,  # 24 hours
                "trade_limit": 1,
                "stop_duration_candles": 48,  # Stop ALL trading for 4 hours
                "max_allowed_drawdown": 0.15,  # Reduced from 25% to 15%
            },
            {
                # LAYER 5: Global stoploss protection
                # If 3 stoplosses occur within 6 hours, stop trading for 4 hours
                "method": "StoplossGuard",
                "lookback_period_candles": 72,  # 6 hours
                "trade_limit": 3,
                "stop_duration_candles": 48,  # Reduced from 144 to 48 (4 hours)
                "only_per_pair": False,  # Global protection
            },
        ]

    # These values can be optimized via Hyperopt
    buy_rsi = IntParameter(15, 60, default=35, space="buy")
    sell_rsi = IntParameter(40, 95, default=70, space="sell")
    buy_mfi = IntParameter(10, 50, default=30, space="buy")
    sell_mfi = IntParameter(50, 90, default=70, space="sell")

    def informative_pairs(self):
        """
        Define additional, macro-level OHLCV data to fetch.
        Auto-detects spot vs futures mode to use the correct pair naming format.
        """
        # Get the current pairs in the whitelist
        pairs = self.dp.current_whitelist()
        # Create a list of tuples containing the pair and the 1h timeframe
        informative_pairs = [(pair, "1h") for pair in pairs]

        # Detect trading mode to use the correct BTC pair name
        # Futures uses BASE/QUOTE:SETTLE format, Spot uses BASE/QUOTE
        is_futures = self.config.get("trading_mode", "spot") == "futures"
        btc_pair = "BTC/USDT:USDT" if is_futures else "BTC/USDT"

        if (btc_pair, "1h") not in informative_pairs:
            informative_pairs.append((btc_pair, "1h"))

        return informative_pairs

    def bot_loop_start(self, **kwargs) -> None:
        """
        Called at the start of every bot iteration.
        Check for global pairlocks and notify via Telegram.
        """
        if not self.dp:
            return

        # Check for global locks (pair == "*") - use filtered helper to skip expired ones
        now_utc = datetime.now(timezone.utc)
        # get_pair_locks handles db vs backtest and filtering by time/active status
        global_locks = PairLocks.get_pair_locks("*", now=now_utc)

        if global_locks:
            # Get the most relevant global lock (the one with the latest end time)
            latest_lock = max(global_locks, key=lambda x: x.lock_end_time)
            lock_reason = latest_lock.reason or "Global protection triggered"
            lock_end = latest_lock.lock_end_time
            
            # Ensure lock_end is naive for timedelta math if it's retrieved as naive from DB
            if lock_end.tzinfo:
                lock_end_utc = lock_end.astimezone(timezone.utc)
            else:
                lock_end_utc = lock_end.replace(tzinfo=timezone.utc)

            # Convert to Local Time (respects TZ environment variable)
            lock_end_local = lock_end_utc.astimezone()
            
            # Format message with DATE to avoid midnight confusion
            date_fmt = "%Y-%m-%d %H:%M:%S"
            msg = f"⛔ GLOBAL LOCK ACTIVE\nReason: {lock_reason}\nUntil: {lock_end_local.strftime(date_fmt)} (Local) / {lock_end_utc.strftime(date_fmt)} (UTC)"  # noqa: E501
            
            # Only send if it's a new lock or different reason
            if msg != self.last_global_lock_msg:
                logger.warning(msg)
                self.dp.send_msg(msg, always_send=True)
                self.last_global_lock_msg = msg
        else:
            # If no global locks exist now, but we had one before, send a recovery message
            if self.last_global_lock_msg:
                msg = "✅ GLOBAL LOCK CLEARED. Bot has resumed trading operations."
                logger.info(msg)
                self.dp.send_msg(msg, always_send=True)
                self.last_global_lock_msg = ""

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        Performance Note: For the best performance, only calculate indicators
        used in entry/exit logic.
        """

        # 1. RSI (Momentum)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # 2. TEMA (Trend tracking with less lag than EMA)
        dataframe["tema"] = ta.TEMA(dataframe, timeperiod=9)

        # 3. Bollinger Bands (Volatility)
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]

        # 4. Volume Metric (Guard)
        dataframe["volume_mean"] = dataframe["volume"].rolling(window=30).mean()
        # Volume spike: current bar volume is > 1.5x the 30-candle average
        dataframe["volume_spike"] = (
            dataframe["volume"] > (dataframe["volume_mean"] * 1.5)
        ).astype(int)

        # 4b. Short-term EMAs for 5m trend detection
        dataframe["ema_20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema_50_5m"] = ta.EMA(dataframe, timeperiod=50)

        # 5. ATR (Average True Range) - Measures volatility for custom_stoploss
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        # 6. Trend Strength & Volatility Metric (For Sideways Exits)
        dataframe["adx"] = ta.ADX(dataframe)

        # 7. MFI (Money Flow Index) - Checks if money is flowing in or out
        dataframe["mfi"] = ta.MFI(dataframe, timeperiod=14)

        # 7b. CMF (Chaikin Money Flow) - Robust volume pressure analysis
        dataframe["cmf"] = ta.ADOSC(
            dataframe["high"],
            dataframe["low"],
            dataframe["close"],
            dataframe["volume"],
            fastperiod=3,
            slowperiod=10,
        )

        # 8. VWAP (Volume Weighted Average Price) - 1 Day rolling (288 candles at 5m)
        dataframe["vwap"] = qtpylib.rolling_vwap(dataframe, window=288)

        # 9. Informative 1h Timeframe
        if not self.dp:
            return dataframe

        inf_tf = "1h"
        # Get the informative dataframe
        informative = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=inf_tf)

        # Calculate 1h RSI, TEMA, and MFI
        informative["rsi"] = ta.RSI(informative, timeperiod=14)
        informative["tema"] = ta.TEMA(informative, timeperiod=9)
        informative["mfi"] = ta.MFI(informative, timeperiod=14)

        # Calculate macro trend lines (Avoid falling knives)
        informative["ema_50"] = ta.EMA(informative, timeperiod=50)
        informative["ema_200"] = ta.EMA(informative, timeperiod=200)

        # Merge the 1h data into our normal 5m dataframe
        dataframe = merge_informative_pair(
            dataframe, informative, self.timeframe, inf_tf, ffill=True
        )
        dataframe["mfi_1h"] = dataframe["mfi_1h"].fillna(50)

        # --- GLOBAL BTC MARKET FILTER ---
        # Get BTC pair name to map its informative columns
        is_futures = self.config.get("trading_mode", "spot") == "futures"
        btc_pair = "BTC/USDT:USDT" if is_futures else "BTC/USDT"

        # Fetch BTC informative data explicitly
        btc_info = self.dp.get_pair_dataframe(pair=btc_pair, timeframe=inf_tf)
        btc_info["btc_rsi_1h"] = ta.RSI(btc_info, timeperiod=14)
        btc_info["btc_ema50_1h"] = ta.EMA(btc_info, timeperiod=50)
        # We need BTC close price to compare with EMA
        btc_info["btc_close_1h"] = btc_info["close"]

        # Determine if BTC is currently dumping (unsafe to buy alts)
        # Safe condition: BTC RSI > 40 OR BTC Price > EMA50
        btc_info["btc_safe"] = (
            (btc_info["btc_rsi_1h"] >= 40) | (btc_info["btc_close_1h"] > btc_info["btc_ema50_1h"])
        ).astype(float)

        # Merge BTC safety status into the current pair's DataFrame
        dataframe = merge_informative_pair(
            dataframe, btc_info[["date", "btc_safe"]], self.timeframe, inf_tf, ffill=True
        )
        # Ensure we don't have NaN safety values block all trades initially (1.0 = True)
        dataframe["btc_safe_1h"] = dataframe["btc_safe_1h"].fillna(1.0)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Multi-signal entry system with tagged entries for performance tracking.
        Signal 1 (Sniper): Original deep dip-buy — rare but highest conviction
        Signal 2 (Scout):  BB bounce in confirmed 1h uptrend — more frequent
        Signal 3 (Reversal): Volume spike capitulation near support — momentum-based
        Each signal has its own enter_tag so you can track which performs best.
        """
        # Initialize columns to prevent NaN/float conversion errors
        dataframe.loc[:, "enter_long"] = 0
        dataframe.loc[:, "enter_short"] = 0
        dataframe.loc[:, "enter_tag"] = ""

        is_spot = self.config.get("trading_mode", "spot") == "spot"

        # Hyperopt-optimizable parameters
        long_rsi_limit = self.buy_rsi.value
        long_mfi_limit = self.buy_mfi.value
        long_rsi_1h_limit = 40 if is_spot else 45

        short_rsi_limit = self.sell_rsi.value
        short_mfi_limit = self.sell_mfi.value

        # --- Volatility Adaptive RSI ---
        dataframe["atr_pct"] = (dataframe["atr"] / dataframe["close"]) * 100
        dataframe["dynamic_buy_rsi"] = long_rsi_limit
        dataframe.loc[dataframe["atr_pct"] > 1.0, "dynamic_buy_rsi"] = long_rsi_limit - 5
        dataframe["dynamic_sell_rsi"] = short_rsi_limit
        dataframe.loc[dataframe["atr_pct"] > 1.0, "dynamic_sell_rsi"] = short_rsi_limit + 5

        # --- SECURITY FILTERS ---
        # 1. Avoid trading in extreme volatility (ATR > 3% of price)
        dataframe["high_volatility"] = (dataframe["atr_pct"] > 3.0).astype(int)
        
        # 2. Require minimum volume (avoid illiquid pairs)
        dataframe["min_volume"] = (dataframe["volume"] > dataframe["volume_mean"] * 0.5).astype(int)
        
        # 3. Combined security condition
        dataframe["secure_to_trade"] = (
            (dataframe["high_volatility"] == 0)  # Not in extreme volatility
            & (dataframe["min_volume"] == 1)  # Has minimum volume
            & (dataframe["btc_safe_1h"] == 1.0)  # BTC is safe
        ).astype(int)

        # ============================================================
        # SIGNAL 1 — SNIPER: Deep dip buy (original strict signal)
        # 7 conditions must align — rare but high confidence.
        # ============================================================
        dataframe.loc[
            (
                (dataframe["secure_to_trade"] == 1)  # Security filter
                & (dataframe["rsi"] < dataframe["dynamic_buy_rsi"])
                & (dataframe["mfi"] < long_mfi_limit)
                & (dataframe["tema"] <= dataframe["bb_middleband"])
                & (dataframe["close"] < dataframe["vwap"])
                & (dataframe["volume"] > 0)
                & (dataframe["rsi_1h"] > long_rsi_1h_limit)
                & (dataframe["adx"] > 15)
                & (dataframe["cmf"] < -0.1)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "sniper_dip")

        # ============================================================
        # SIGNAL 2 — SCOUT: Bollinger Band bounce in confirmed uptrend
        # Catches healthy pullbacks in coins trending up on 1h.
        # ============================================================
        dataframe.loc[
            (
                (dataframe["secure_to_trade"] == 1)  # Security filter
                & (dataframe["enter_long"] == 0)  # Don't override sniper
                & (dataframe["close"] <= dataframe["bb_lowerband"] * 1.015)  # At/near lower BB
                & (dataframe["ema_50_1h"] > dataframe["ema_200_1h"])  # 1h uptrend confirmed
                & (dataframe["rsi"] < 45)  # Moderately oversold
                & (dataframe["volume"] > 0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "bb_bounce_uptrend")

        # ============================================================
        # SIGNAL 3 — REVERSAL: Volume spike capitulation near support
        # Sharp selloffs with huge volume often snap back.
        # ============================================================
        dataframe.loc[
            (
                (dataframe["secure_to_trade"] == 1)  # Security filter
                & (dataframe["enter_long"] == 0)  # Don't override previous signals
                & (dataframe["rsi"] < 35)  # Oversold
                & (dataframe["volume_spike"] == 1)  # Big volume = capitulation
                & (dataframe["close"] < dataframe["bb_middleband"])  # Below BB mid
                & (dataframe["close"] > dataframe["bb_lowerband"])  # Not in freefall
                & (dataframe["ema_50_1h"] > dataframe["ema_200_1h"])  # 1h uptrend
                & (dataframe["volume"] > 0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "volume_reversal")

# ============================================================
        # SIGNAL 4 — MOMENTUM: Breakout in confirmed uptrend
        # Catches coins pumping with strong volume.
        # ============================================================
        dataframe.loc[
            (
                (dataframe["secure_to_trade"] == 1)  # Security filter
                & (dataframe["enter_long"] == 0)  # Don't override previous signals
                & (dataframe["close"] > dataframe["vwap"])  # Above VWAP (strength)
                & (dataframe["close"] > dataframe["ema_50_1h"])  # Price above 1h EMA50
                & (dataframe["close"] > dataframe["bb_middleband"])  # Above BB mid (momentum)
                & (dataframe["rsi"] > 55)  # Clear momentum
                & (dataframe["rsi"] < 80)  # Not fully exhausted (pump RSI is 70-80)
                & (dataframe["adx"] > 20)  # Trending (not sideways chop)
                & (dataframe["volume"] > dataframe["volume_mean"] * 1.5)  # Require strong volume spike
                & (dataframe["rsi_1h"] > 45)  # 1h trend strength
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "momentum_breakout")

        # ============================================================
        # SIGNAL 5 — TREND FOLLOW: REMOVED - 4/4 losses
        # Original trend entry was too permissive. Removed to prevent losses.
        # ============================================================

        # ============================================================
        # SIGNAL 6 — SIMPLE ENTRY: Market entry with fewer conditions
        # Fallback signal for when no other signals trigger.
        # ============================================================
        dataframe.loc[
            (
                (dataframe["secure_to_trade"] == 1)  # Security filter
                & (dataframe["enter_long"] == 0)  # Only if no other signal fired
                & (dataframe["rsi"] < 45)  # Not overbought
                & (dataframe["close"] > dataframe["ema_20"])  # Above short-term trend
                & (dataframe["close"] > dataframe["ema_50_1h"])  # Above 1h trend
                & (dataframe["rsi_1h"] > 45)  # 1h trend strength
                & (dataframe["volume"] > dataframe["volume_mean"] * 1.2)  # Volume filter
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "simple_entry")

        # --- SHORT Entry: Adaptive RSI + MFI peak + VWAP Premium ---
        dataframe.loc[
            (
                (dataframe["rsi"] > dataframe["dynamic_sell_rsi"])
                & (dataframe["mfi"] > short_mfi_limit)
                & (dataframe["tema"] >= dataframe["bb_middleband"])
                & (dataframe["close"] > dataframe["vwap"])
                & (dataframe["volume"] > 0)
                & (dataframe["rsi_1h"] < 60)
                & (dataframe["adx"] > 15)
                & (dataframe["cmf"] > 0.1)
            ),
            ["enter_short", "enter_tag"],
        ] = (1, "rsi_mfi_vwap_peak")

        # --- SIGNAL DIAGNOSTIC ---
        # Freqtrade evaluates iloc[-2] (last CLOSED candle) for entry signals,
        # not iloc[-1] (currently forming candle). Log both to diagnose.
        for label, idx in [("CLOSED[-2]", -2), ("LIVE[-1]", -1)]:
            c = dataframe.iloc[idx]
            fired = int(c.get("enter_long", 0))
            tag = c.get("enter_tag", "") if fired else "none"
            if fired == 0 and label == "CLOSED[-2]":
                logger.debug(
                    f"📊 {metadata['pair']} {label} enter={fired} tag={tag} | "
                    f"RSI={c.get('rsi', 0):.1f} ADX={c.get('adx', 0):.1f} "
                    f"ema20={c.get('ema_20', 0):.4f} close={c['close']:.4f} "
                    f"ema50_1h={c.get('ema_50_1h', 0):.4f} bb_mid={c.get('bb_middleband', 0):.4f} "
                    f"vwap={c.get('vwap', 0):.4f} btc_safe={c.get('btc_safe_1h', 'MISSING')}"
                )
            elif fired == 1:
                logger.info(f"🎯 {metadata['pair']} {label} SIGNAL FIRED: {tag}")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Signal-aware exit conditions:
        - Dip-buy signals (sniper_dip, bb_bounce, volume_reversal): exit on RSI overbought / BB upper
        - Momentum signals (momentum_breakout, trend_follow): exit when momentum FADES
          (RSI drops below 50, or price crosses under EMA20)
        Freqtrade uses the FIRST matching exit condition, so we set the most specific ones first.
        """
        # Initialize columns to prevent NaN/float conversion errors
        dataframe.loc[:, "exit_long"] = 0
        dataframe.loc[:, "exit_short"] = 0
        dataframe.loc[:, "exit_tag"] = ""

        # --- EXIT LONG: Momentum fading (RSI < 50) ---
        dataframe.loc[
            (dataframe["rsi"] < 50),
            ["exit_long", "exit_tag"],
        ] = (1, "rsi_momentum_fade")

        # --- EXIT LONG: Price crosses under EMA20 - Only trigger when price < EMA20 * 0.995 (avoid minor dips)
        # AND RSI < 45 to avoid triggering on sideways movement
        dataframe.loc[
            (
                (dataframe["close"] < dataframe["ema_20"] * 0.995)
                & (dataframe["rsi"] < 45)
            ),
            ["exit_long", "exit_tag"],
        ] = (1, "price_ema20_cross_below")

        # --- EXIT LONG: Price above BB upper (take profit) ---
        # Exit when price > BB upper with bullish candle
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["bb_upperband"])
                & (dataframe["close"] > dataframe["open"])
            ),
            ["exit_long", "exit_tag"],
        ] = (1, "bb_upper_take_profit")

        # --- EXIT LONG: Extreme overbought RSI ---
        dataframe.loc[
            (qtpylib.crossed_above(dataframe["rsi"], self.sell_rsi.value)),
            ["exit_long", "exit_tag"],
        ] = (1, "extreme_overbought")

        # --- EXIT SHORT: RSI oversold or price below lower Bollinger Band ---
        dataframe.loc[
            (
                qtpylib.crossed_below(dataframe["rsi"], self.buy_rsi.value)
                | (dataframe["tema"] < dataframe["bb_lowerband"])
            ),
            ["exit_short", "exit_tag"],
        ] = (1, "cover_short_signal")

        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        """
        Custom exit algorithm. Overrides normal stoploss and ROI.
        Allows us to dynamically escape sideways markets or "unclog" bad trades.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None
        last_candle = dataframe.iloc[-1].squeeze()

        # How many minutes has this trade been open?
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 60

        # --- TIME-BASED FORCED EXIT (Stuck trades) ---
        # Exit if stuck for 2+ hours and not profitable
        if trade_duration > 120 and current_profit < 0.005:
            return "forced_exit_stuck_2h"

        # Exit if stuck for 4+ hours regardless of profit
        if trade_duration > 240:
            return "forced_exit_4h_max"

        # --- PROFIT TARGET EXIT ---
        # Exit if RSI is overbought (>70) and profit > 0.5%
        if last_candle["rsi"] > 70 and current_profit > 0.005:
            return "take_profit_rsi_overbought"
        
        # Exit if price is above BB upper band with >0.5% profit
        if current_rate > last_candle["bb_upperband"] and current_profit > 0.005:
            return "take_profit_bb_upper"
        
        # Take-profit for momentum_breakout trades at 0.8% target
        if trade.enter_tag == "momentum_breakout" and current_profit > 0.008:
            return "momentum_take_profit_08"

        # --- BULL TRAP EXIT (Momentum trades failing quickly) ---
        if trade.enter_tag == "momentum_breakout" and trade_duration < 15:
            # If RSI drops below 50 within 15 mins of entry, exit immediately
            if last_candle["rsi"] < 50:
                return "momentum_bull_trap_exit"

        # --- Tag-Specific Exit Logic ---
        momentum_tags = ["momentum_breakout", "simple_entry"]
        dip_buy_tags = ["sniper_dip", "bb_bounce", "volume_reversal"]

        if trade.enter_tag in momentum_tags:
            # EXIT LONG (Momentum trades): momentum fading only on losses > -0.3%
            if last_candle["rsi"] < 40 and current_profit < -0.003:
                return "momentum_fade_rsi_drop"
            if current_rate < last_candle["ema_20"] * 0.995 and last_candle["rsi"] < 45 and current_profit < -0.003:
                return "momentum_fade_ema20_drop"

        elif trade.enter_tag in dip_buy_tags:
            # EXIT LONG (Dip-buy trades): RSI overbought or price above upper BB
            if last_candle["rsi"] >= self.sell_rsi.value:
                return "take_profit_overbought"
            if current_rate > last_candle["bb_upperband"]:
                return "take_profit_bb_upper"

        # --- TIME-BASED EXITS (Capital Free-up) ---
        # If the trade has been stuck for 24+ hours...
        if trade_duration > 1440:
            if current_profit >= 0.005:
                return "stagnant_profitable_24h"
            if trade_duration > 2880:
                return "surrender_stagnant_48h"

        return None

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        """
        PHASE 3: ATR-based adaptive and dynamic stoploss.
        Replaces flat percentage stoploss. Uses a multiple of ATR instead.
        Calculations must result in a percentage relative to the current rate as per Freqtrade docs.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        last_candle = dataframe.iloc[-1]

        # Use 2.0x the Average True Range as the acceptable volatility cushion (increased from 1.5x)
        atr_value = last_candle["atr"]
        atr_stop_distance = atr_value * 2.0
        # Minimum 1% buffer to avoid catching normal volatility
        stoploss_pct = atr_stop_distance / current_rate
        if stoploss_pct < 0.01:
            stoploss_pct = 0.01
        # We must calculate what percentage that ATR distance is from the current asset price
        # E.g. If BTC is 60000 and ATR is 1000, 1.5 * 1000 = 1500 distance.
        # 1500 / 60000 = 2.5%
        stoploss_pct = min(stoploss_pct, 0.15)

        if trade.is_short:
            return (
                -stoploss_pct
            )  # For short, Freqtrade expects a negative stoploss ratio relative to current_rate
        else:
            return -stoploss_pct  # For long, it also expects a negative stoploss ratio

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> float | None | tuple[float | None, str | None]:
        """
        Layered DCA: Rải vốn linh hoạt ở mức -4%, -8%, -12% với khối lượng Martingale.
        This provides a 3-layer safety net to rescue trades caught in deep pullbacks.
        """
        # Don't try to DCA if there are already open orders on this trade
        if trade.has_open_orders:
            return None

        # Maximum of 3 safety orders (+1 initial = 4 total entries)
        max_dca_orders = 3
        count_of_entries = trade.nr_of_successful_entries

        if count_of_entries > max_dca_orders:
            return None

        # Get latest candle data for the pair to check momentum
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe.empty:
            return None

        last_candle = dataframe.iloc[-1].squeeze()

        # DCA Levels Config: (Level, Profit threshold, Stake Multiplier)
        dca_config = [
            (1, -0.04, 1.0),  # Level 1: at -4%, buy 1.0x the original stake
            (2, -0.08, 1.5),  # Level 2: at -8%, buy 1.5x the original stake
            (3, -0.12, 2.0),  # Level 3: at -12%, buy 2.0x the original stake
        ]

        expected_entries, profit_threshold, stake_multiplier = dca_config[count_of_entries - 1]

        if current_profit < profit_threshold:
            # Guard: Block DCA entirely for momentum/trend entries.
            # If a momentum trade reverses, it means the trend broke — adding money
            # to a failed momentum trade compounds losses. Cut, don't average down.
            momentum_tags = {"momentum_breakout", "trend_follow"}
            if trade.enter_tag in momentum_tags:
                logger.info(
                    f"Skipping DCA for {trade.pair}: momentum entry ({trade.enter_tag}) "
                    f"does not support averaging down."
                )
                return None

            # Guard: only allow deeper DCA (level 2+) if 1h trend is still up.
            # Prevents throwing good money after bad in a genuine crash.
            is_uptrend_1h = last_candle.get("ema_50_1h", 0) > last_candle.get("ema_200_1h", 0)
            if count_of_entries >= 2 and not is_uptrend_1h:
                logger.info(
                    f"Skipping DCA level {count_of_entries} for {trade.pair}: "
                    f"1h downtrend (EMA50 < EMA200)"
                )
                return None

            # Check momentum so we don't catch a falling knife right at the dump peak
            # Wait for RSI or MFI to be reasonably oversold to ensure a bounce
            if last_candle["rsi"] < 35 or last_candle["mfi"] < 35:
                # Calculate the new stake amount based on the original base stake
                filled_entries = trade.select_filled_orders(trade.entry_side)
                if filled_entries:
                    base_stake = filled_entries[0].stake_amount_filled
                    dca_stake = base_stake * stake_multiplier
                    return min(dca_stake, max_stake), f"dca_level_{count_of_entries}"

        return None

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str,
        side: str,
        **kwargs,
    ) -> bool:
        """
        Called right before placing a buy/short order.
        We use Google Gemini LLM to act as a final "sanity check" for the trade.
        """
        # 1. Backtesting Guard: NEVER call APIs during backtesting/hyperopt.
        if self.dp.runmode.value in ("backtest", "hyperopt"):
            return True

        if not getattr(self, "llm_enabled", False):
            return True

        try:
            # 2. Extract recent data
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty:
                return True

            # Get the last 10 closed candles for context
            last_candles = dataframe.tail(10)
            latest = last_candles.iloc[-1]

            # --- DAILY BUDGET CHECK ---
            today_str = str(current_time.date())
            if self.ai_budget_date != today_str:
                self.ai_daily_calls = 0
                self.ai_budget_date = today_str
                self._save_gemini_usage()

            if self.ai_daily_calls >= self.ai_daily_budget:
                msg = f"💸 Daily AI budget ({self.ai_daily_budget} calls) reached. Approving remaining trades without AI."  # noqa: E501
                if self.ai_daily_calls == self.ai_daily_budget: # Only warn once
                    logger.warning(msg)
                    self.dp.send_msg(msg, always_send=True)
                    self.ai_daily_calls += 1 # bump so we don't spam the warning
                    self._save_gemini_usage()
                return True

            # --- DECISION CACHE CHECK (COST REDUCTION) ---
            # If we asked Gemini about this exact pair + side + strategy signal 
            # within the last 15 minutes (3 candles), reuse the decision to save API cost.
            candle_timestamp = latest.name.timestamp() if hasattr(latest.name, "timestamp") else current_time.timestamp()
            cache_key = (pair, side, entry_tag)
            
            if cache_key in self.ai_candle_cache:
                cached_decision, cached_time = self.ai_candle_cache[cache_key]
                # Check if the cache is still valid (less than 900 seconds / 15 mins old)
                if (candle_timestamp - cached_time) < 900:
                    logger.info(
                        f"📦 Reusing Cached AI decision for {pair} ({entry_tag}): {'APPROVED' if cached_decision else 'REJECTED'}"
                    )
                    return cached_decision

            self.ai_daily_calls += 1
            self._save_gemini_usage()
            
            msg = f"🧠 Asking Gemini AI [{self.ai_daily_calls}/{self.ai_daily_budget} today] to analyze {side} on {pair} at {rate}..."  # noqa: E501
            logger.info(msg)
            
            # Format inputs precisely to save tokens
            fmt = "{:.2f}"
            current_rsi_1h = fmt.format(latest.get('rsi_1h', 0))
            current_ema50_1h = fmt.format(latest.get('ema_50_1h', 0))
            current_ema200_1h = fmt.format(latest.get('ema_200_1h', 0))
            current_atr = fmt.format(latest.get('atr', 0))
            current_vwap = fmt.format(latest.get('vwap', 0))
            btc_safe = int(latest.get('btc_safe_1h', 1.0))

            prompt = f"Trade: {side} @ {rate}\n"
            prompt += f"Strategy Signal: {entry_tag}\n"
            prompt += f"Context: VWAP={current_vwap} | 1hRSI={current_rsi_1h} | 1hEMA50/200={current_ema50_1h}/{current_ema200_1h} | ATR={current_atr} | BTC_Safe={btc_safe}\n\n"  # noqa: E501
            prompt += "Last 10 Candles (5m):\n"
            prompt += "Close | Volume | RSI | MFI | ADX | TEMA | BB_Mid\n"
            for _, row in last_candles.iterrows():
                prompt += f"{fmt.format(row['close'])} | {fmt.format(row['volume'])} | {fmt.format(row['rsi'])} | {fmt.format(row['mfi'])} | {fmt.format(row['adx'])} | {fmt.format(row['tema'])} | {fmt.format(row['bb_middleband'])}\n"  # noqa: E501

            response = self.llm_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are a quant. Analyze technical indicators and market structure. decision=True if it is a strong setup, False if it is a trap. Set confidence (0-100). reasoning=max 15 words.",  # noqa: E501
                    response_mime_type="application/json",
                    response_schema=TradeDecision,
                    temperature=0.1,
                ),
            )

            # The SDK will automatically validate the JSON output against the Pydantic model
            result = json.loads(response.text)

            confidence = result.get("confidence", 0)
            # Lowered from 65 to 55 - better trade capture rate
            if result.get("decision") is True and confidence >= 55:
                result_msg = f"✅ Gemini APPROVED {side} on {pair} (Confidence: {confidence}%). {result.get('reasoning')}"  # noqa: E501
                logger.info(result_msg)
                self.dp.send_msg(result_msg, always_send=True)
                self.ai_candle_cache[cache_key] = (True, candle_timestamp)
                # Prune old cache entries (keep max 200 entries)
                if len(self.ai_candle_cache) > 200:
                    oldest = next(iter(self.ai_candle_cache))
                    del self.ai_candle_cache[oldest]
                return True
            else:
                decision_str = "REJECTED" if not result.get("decision") else "LOW CONFIDENCE"
                result_msg = f"❌ Gemini {decision_str} {side} on {pair} (Confidence: {confidence}%). {result.get('reasoning')}"  # noqa: E501
                logger.info(result_msg)
                self.ai_candle_cache[cache_key] = (False, candle_timestamp)
                return False

        except Exception as e:
            msg = f"⚠️ Error calling Gemini API: {e}. Defaulting to YES to keep bot running."
            logger.error(msg)
            return True

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        """
        Called right before placing a sell/exit order.
        Use Google Gemini LLM to verify if a winning trade is experiencing a massive breakout.
        If it is, AI will reject the normal ROI exit so we can "let winners run".
        """
        # Record trade result for signal performance tracking BEFORE AI check
        if exit_reason in ["roi", "trailing_stop_profit", "exit_signal"]:
            current_profit = trade.calc_profit_ratio(rate)
            self.record_trade_result(trade.enter_tag, current_profit)
        
        # 1. Backtesting Guard & Enabled Check
        if self.dp.runmode.value in ("backtest", "hyperopt") or not getattr(
            self, "llm_enabled", False
        ):
            return True

        # We ONLY want AI to intervene if the strategy is trying to take profit normally (roi or trailing)  # noqa: E501
        # We DO NOT override stoplosses! If we're bleeding, get out!
        if exit_reason not in ["roi", "trailing_stop_profit", "exit_signal"]:
            return True

        # Only intervene if the trade is actually in meaningful profit (> 1.5%)
        # This prevents AI spam on minor trailing stop exits.
        current_profit = trade.calc_profit_ratio(rate)
        if current_profit <= 0.015:
            return True

        # Avoid asking AI again for the same trade multiple times in quick succession
        cache_key = f"exit_{trade.id}_{current_profit:.3f}"
        if getattr(self, "_recent_exit_checks", None) is None:
            self._recent_exit_checks = set()
        if cache_key in self._recent_exit_checks:
            return True
        self._recent_exit_checks.add(cache_key)
        if len(self._recent_exit_checks) > 100:
            self._recent_exit_checks.clear()

        try:
            # 2. Extract recent data for the exit analysis
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty:
                return True

            last_candles = dataframe.tail(10)

            # Format inputs precisely to save tokens
            fmt = "{:.2f}"
            prompt = f"OPEN: {trade.trade_direction} | Profit: {current_profit * 100:.2f}% | Exit Reason: {exit_reason}\n\n"  # noqa: E501
            prompt += "Last 10 Candles (5m):\n"
            prompt += "Close | Volume | RSI | MFI | ADX | mfi_1h\n"
            for _, row in last_candles.iterrows():
                prompt += f"{fmt.format(row['close'])} | {fmt.format(row['volume'])} | {fmt.format(row['rsi'])} | {fmt.format(row['mfi'])} | {fmt.format(row['adx'])} | {fmt.format(row.get('mfi_1h', 50))}\n"  # noqa: E501

            msg = f"💎 Bot wants to take profit ({current_profit * 100:.2f}%) on {pair} at {rate}. Asking Gemini if we should hold the breakout..."  # noqa: E501
            logger.info(msg)

            response = self.llm_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are a quant. If violent breakout/pump continuing, set hold_trade=True to let winners run. If momentum dying/exhausted, hold_trade=False. reasoning=max 15 words.",  # noqa: E501
                    response_mime_type="application/json",
                    response_schema=ExitDecision,
                    temperature=0.1,
                ),
            )

            result = json.loads(response.text)

            if result.get("hold_trade") is True:
                logger.info(
                    f"🚀 Gemini overrides profit exit! HOLDING {pair} for a bigger pump! Reasoning: {result.get('reasoning')}"  # noqa: E501
                )
                return False  # Reject exit, keep trade open
            else:
                logger.info(
                    f"💰 Gemini approves locking in profit for {pair}. Reasoning: {result.get('reasoning')}"  # noqa: E501
                )
                return True  # Approve exit

        except Exception as e:
            msg = f"⚠️ Error calling Gemini API on exit: {e}. Selling to be safe."
            logger.error(msg)
            return True
