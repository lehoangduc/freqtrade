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
from freqtrade.persistence import Trade, Order
from datetime import datetime
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
    decision: bool = Field(
        description="Set to True if the trade setup is highly probable / safe. Set to False if this is likely a trap, dump, or poor setup."  # noqa: E501
    )
    confidence: int = Field(
        description="Confidence score from 0 to 100 for this decision. Above 65 is required for entry."  # noqa: E501
    )
    reasoning: str = Field(description="A concise 1-sentence technical reason for your decision.")


class ExitDecision(BaseModel):
    hold_trade: bool = Field(
        description="Set to True to REJECT the exit order and HOLD for more profit because a massive breakout is occurring. Set to False to ALLOW selling and take the money now."  # noqa: E501
    )
    reasoning: str = Field(
        description="A concise 1-sentence analytical reason why we should hold or sell right now."
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

        # Configure Google Generative AI for trade confirmation
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            self.llm_client = genai.Client(api_key=api_key)
            self.llm_enabled = True
            # Cache: {(pair, candle_timestamp) -> bool} - ONE API call per candle per pair
            self.ai_candle_cache: dict = {}
            # Daily budget: track calls to cap max spend
            self.ai_daily_calls = 0
            self.ai_budget_date = None
            self.ai_daily_budget = 100  # Max 100 AI calls per day (~$0.002 total)
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

    # Trailing stoploss — lock in profit at 1% once we're 2% up
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Enable ATR-based custom stoploss
    use_custom_stoploss = True

    # Enable DCA (Dollar Cost Averaging) - automatically buy more on dips
    position_adjustment_enable = True

    # Process indicators only for new candles to save compute
    process_only_new_candles = True

    @property
    def protections(self):
        """
        Advanced Layered Protections for Active Trading:
        Isolates "toxic" performing coins without shutting down the entire bot.
        """
        return [
            {
                # LAYER 1: The "Breather"
                # Wait 5 candles (25 mins) before trying to trade a coin we just exited.
                "method": "CooldownPeriod",
                "stop_duration_candles": 5,
            },
            {
                # LAYER 2: The "Toxic Pair" filter
                # If a specific coin hits its stoploss 2 times in the last 4 hours...
                "method": "StoplossGuard",
                "lookback_period_candles": 48,  # 4 hours
                "trade_limit": 2,
                "stop_duration_candles": 24,  # ...ban THAT specific coin for 2 hours.
                "only_per_pair": True,
            },
            {
                # LAYER 3: The "Loser" filter
                # If a pair has lost 3 trades in a row recently...
                "method": "LowProfitPairs",
                "lookback_period_candles": 144,  # 12 hours
                "trade_limit": 3,
                "stop_duration_candles": 48,  # ...block it for 4 hours (market makers are playing us).  # noqa: E501
                "required_profit": 0.0,
            },
            {
                # LAYER 4: The "Flash Crash" global panic switch
                # If the bot's total wallet drops by 10% in a 24 hour period...
                "method": "MaxDrawdown",
                "lookback_period_candles": 288,  # 24 hours
                "trade_limit": 1,
                "stop_duration_candles": 24,  # ...Stop ALL trading for 2 hours to weather the storm.  # noqa: E501
                "max_allowed_drawdown": 0.10,
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

        # ============================================================
        # SIGNAL 1 — SNIPER: Deep dip buy (original strict signal)
        # 8 conditions must align — rare but high confidence.
        # ============================================================
        dataframe.loc[
            (
                (dataframe["rsi"] < dataframe["dynamic_buy_rsi"])
                & (dataframe["mfi"] < long_mfi_limit)
                & (dataframe["tema"] <= dataframe["bb_middleband"])
                & (dataframe["close"] < dataframe["vwap"])
                & (dataframe["volume"] > 0)
                & (dataframe["rsi_1h"] > long_rsi_1h_limit)
                & (dataframe["adx"] > 15)
                & (dataframe["cmf"] < -0.1)
                & (dataframe["btc_safe_1h"] == 1.0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "sniper_dip")

        # ============================================================
        # SIGNAL 2 — SCOUT: Bollinger Band bounce in confirmed uptrend
        # Catches healthy pullbacks in coins trending up on 1h.
        # Less strict than sniper — the 1h uptrend IS the conviction.
        # ============================================================
        dataframe.loc[
            (
                (dataframe["enter_long"] == 0)  # Don't override sniper
                & (dataframe["close"] <= dataframe["bb_lowerband"] * 1.01)  # At/near lower BB
                & (dataframe["ema_50_1h"] > dataframe["ema_200_1h"])  # 1h uptrend confirmed
                & (dataframe["rsi"] < 40)  # Moderately oversold (not extreme required)
                & (dataframe["volume_spike"] == 1)  # Volume confirms the move
                & (dataframe["volume"] > 0)
                & (dataframe["btc_safe_1h"] == 1.0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "bb_bounce_uptrend")

        # ============================================================
        # SIGNAL 3 — REVERSAL: Volume spike capitulation near support
        # Sharp selloffs with huge volume often snap back.
        # Requires 1h uptrend to avoid catching true crashes.
        # ============================================================
        dataframe.loc[
            (
                (dataframe["enter_long"] == 0)  # Don't override previous signals
                & (dataframe["rsi"] < 35)  # Oversold
                & (dataframe["volume_spike"] == 1)  # Big volume = capitulation
                & (dataframe["close"] < dataframe["bb_middleband"])  # Below BB mid
                & (dataframe["close"] > dataframe["bb_lowerband"])  # Not in freefall
                & (dataframe["ema_50_1h"] > dataframe["ema_200_1h"])  # 1h uptrend
                & (dataframe["mfi"] < 40)  # Money flowing out (capitulation)
                & (dataframe["volume"] > 0)
                & (dataframe["btc_safe_1h"] == 1.0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "volume_reversal")

        # ============================================================
        # SIGNAL 4 — MOMENTUM: Breakout buy in strong trend
        # Catches coins that are ALREADY pumping. Instead of buying
        # dips, we ride the wave. The Gemini AI guard is the key
        # filter here — it prevents buying exhausted pumps.
        # Conditions: price breaking above BB upper + strong volume
        # + confirmed 1h uptrend + RSI in the sweet spot (not yet
        # overbought/exhausted).
        # ============================================================
        dataframe.loc[
            (
                (dataframe["enter_long"] == 0)  # Don't override previous signals
                & (dataframe["close"] > dataframe["bb_upperband"])  # Breaking out above BB
                & (dataframe["close"] > dataframe["vwap"])  # Above VWAP (strength)
                & (dataframe["ema_50_1h"] > dataframe["ema_200_1h"])  # 1h uptrend
                & (dataframe["rsi"] > 50)  # Momentum present
                & (dataframe["rsi"] < 70)  # But not exhausted yet
                & (dataframe["volume_spike"] == 1)  # Volume confirms breakout
                & (dataframe["adx"] > 25)  # Strong trend (not sideways chop)
                & (dataframe["volume"] > 0)
                & (dataframe["btc_safe_1h"] == 1.0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "momentum_breakout")

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

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate exit conditions for LONG and SHORT trades.
        """
        # --- EXIT LONG: RSI overbought or price above upper Bollinger Band ---
        dataframe.loc[
            (
                (
                    qtpylib.crossed_above(dataframe["rsi"], self.sell_rsi.value)
                )  # Signal: RSI overbought
                | (dataframe["tema"] > dataframe["bb_upperband"])  # Signal: Price above upper BB
            ),
            ["exit_long", "exit_tag"],
        ] = (1, "take_profit_signal")

        # --- EXIT SHORT: RSI oversold or price below lower Bollinger Band ---
        dataframe.loc[
            (
                (
                    qtpylib.crossed_below(dataframe["rsi"], self.buy_rsi.value)
                )  # Signal: RSI oversold (short covering)
                | (dataframe["tema"] < dataframe["bb_lowerband"])  # Signal: Price below lower BB
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
        last_candle = dataframe.iloc[-1].squeeze()

        # How many minutes has this trade been open?
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 60

        # PHASE 3: TIME-BASED EXITS (Capital Free-up)
        # If the trade has been stuck and flat for 24 hours (1440 minutes)...
        if trade_duration > 1440:
            # If we are slightly profitable or breakeven, exit immediately to free up money
            if current_profit >= 0.005:
                return "stagnant_profitable_24h"
            # If we are losing money, wait up to 48 hours (2880 mins) before surrendering it
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

        # Use 2.5x the Average True Range as the acceptable volatility cushion
        atr_value = last_candle["atr"]
        atr_stop_distance = atr_value * 2.5

        # We must calculate what percentage that ATR distance is from the current asset price
        # E.g. If BTC is 60000 and ATR is 1000, 2.5 * 1000 = 2500 distance.
        # 2500 / 60000 = 4.16%
        stoploss_pct = atr_stop_distance / current_rate

        # Return negative value for long stoploss, positive for short stoploss
        # Limit the maximum absolute stoploss to 15% to prevent catastrophic dumps
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
            today = current_time.date()
            if self.ai_budget_date != today:
                self.ai_daily_calls = 0
                self.ai_budget_date = today
            if self.ai_daily_calls >= self.ai_daily_budget:
                msg = f"💸 Daily AI budget ({self.ai_daily_budget} calls) reached. Approving remaining trades without AI."  # noqa: E501
                logger.warning(msg)
                self.dp.send_msg(msg)
                return True

            # --- CANDLE CACHE CHECK ---
            candle_ts = latest.name  # DatetimeIndex
            cache_key = (pair, candle_ts)
            if cache_key in self.ai_candle_cache:
                cached = self.ai_candle_cache[cache_key]
                logger.info(
                    f"📦 Cached AI decision for {pair}: {'APPROVED' if cached else 'REJECTED'}"
                )
                return cached

            self.ai_daily_calls += 1
            msg = f"🧠 Asking Gemini AI [{self.ai_daily_calls}/{self.ai_daily_budget} today] to analyze {side} on {pair} at {rate}..."  # noqa: E501
            logger.info(msg)
            self.dp.send_msg(msg, always_send=True)
            current_rsi_1h = f"{latest.get('rsi_1h', 0):.1f}"
            current_ema50_1h = f"{latest.get('ema_50_1h', 0):.4f}"
            current_ema200_1h = f"{latest.get('ema_200_1h', 0):.4f}"
            current_atr = f"{latest.get('atr', 0):.4f}"
            current_vwap = f"{latest.get('vwap', 0):.4f}"

            prompt = f"Trade: {side} @ {rate}\n"
            prompt += f"Context: VWAP={current_vwap} | 1hRSI={current_rsi_1h} | 1hEMA50/200={current_ema50_1h}/{current_ema200_1h} | ATR={current_atr}\n\n"  # noqa: E501
            prompt += "Last 10 Candles (5m):\n"
            prompt += "Close | Volume | RSI | MFI | ADX | TEMA | BB_Mid\n"
            for _, row in last_candles.iterrows():
                prompt += f"{row['close']:.4f} | {row['volume']:.2f} | {row['rsi']:.1f} | {row['mfi']:.1f} | {row['adx']:.1f} | {row['tema']:.4f} | {row['bb_middleband']:.4f}\n"  # noqa: E501

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
            if result.get("decision") is True and confidence >= 65:
                result_msg = f"✅ Gemini APPROVED {side} on {pair} (Confidence: {confidence}%). {result.get('reasoning')}"  # noqa: E501
                logger.info(result_msg)
                self.dp.send_msg(result_msg, always_send=True)
                self.ai_candle_cache[cache_key] = True
                # Prune old cache entries (keep max 200 entries)
                if len(self.ai_candle_cache) > 200:
                    oldest = next(iter(self.ai_candle_cache))
                    del self.ai_candle_cache[oldest]
                return True
            else:
                decision_str = "REJECTED" if not result.get("decision") else "LOW CONFIDENCE"
                result_msg = f"❌ Gemini {decision_str} {side} on {pair} (Confidence: {confidence}%). {result.get('reasoning')}"  # noqa: E501
                logger.info(result_msg)
                self.dp.send_msg(result_msg, always_send=True)
                self.ai_candle_cache[cache_key] = False
                return False

        except Exception as e:
            msg = f"⚠️ Error calling Gemini API: {e}. Defaulting to YES to keep bot running."
            logger.error(msg)
            self.dp.send_msg(msg)
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

            prompt = f"OPEN: {trade.trade_direction} | Profit: {current_profit * 100:.2f}% | Exit Reason: {exit_reason}\n\n"  # noqa: E501
            prompt += "Last 10 Candles (5m):\n"
            prompt += "Close | Volume | RSI | MFI | ADX | 1hMFI\n"
            for _, row in last_candles.iterrows():
                prompt += f"{row['close']:.4f} | {row['volume']:.2f} | {row['rsi']:.1f} | {row['mfi']:.1f} | {row['adx']:.1f} | {row.get('mfi_1h', 50):.1f}\n"  # noqa: E501

            msg = f"💎 Bot wants to take profit ({current_profit * 100:.2f}%) on {pair} at {rate}. Asking Gemini if we should hold the breakout..."  # noqa: E501
            logger.info(msg)
            self.dp.send_msg(msg)

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
            self.dp.send_msg(msg)
            return True
