#!/usr/bin/env python3
import os
import sys
from datetime import datetime, timezone


sys.path.insert(0, os.path.dirname(__file__))

from freqtrade.configuration import Configuration
from freqtrade.persistence import Order, Trade, init_db
from freqtrade.resolvers import ExchangeResolver


config = Configuration.from_files(["user_data/config.json", "user_data/config_spot.json"])
config["exchange"]["sandbox"] = False

# Initialize the db
db_url = config.get("db_url", "sqlite:///user_data/tradesv3-spot.sqlite")
init_db(db_url)

exchange = ExchangeResolver.load_exchange(config)
balances = exchange.get_balances()

# Find existing open trades
open_trades_query = Trade.get_open_trades()
open_trade_coins = [trade.pair.split("/")[0] for trade in open_trades_query]

print("=== Scanning Exchange Wallet for Balances ===")

for coin, info in balances.items():
    # ccxt balances also returns metadata like 'info', 'free', 'used', 'total' at the top level
    if not isinstance(info, dict):
        continue

    if coin in ["USDT", "USDC"]:
        continue

    amount = info.get("total", 0)
    if amount <= 0:
        continue

    pair = f"{coin}/USDT"

    # Needs a price to know if it's dust or not
    try:
        ticker = exchange.fetch_ticker(pair)
        current_price = ticker["last"]
    except Exception as e:
        print(f"Skipping {pair} (Could not fetch ticker: {e})")
        continue

    value_usdt = amount * current_price

    # Ignore dust (< $5)
    if value_usdt < 5.0:
        continue

    # Check if already in Freqtrade DB
    if coin in open_trade_coins:
        print(
            f"✅ {pair}: Already tracked by Freqtrade DB (Amt: {amount:.4f}, Value: ${value_usdt:.2f})"
        )
        continue

    print(
        f"⚠️ {pair}: Found in wallet but NOT in Freqtrade DB! Amt: {amount:.4f}, Value: ${value_usdt:.2f}"
    )

    # We will simulate a trade entry right now at the current price
    trade = Trade(
        pair=pair,
        base_currency=coin,
        stake_currency="USDT",
        stake_amount=value_usdt,
        amount=amount,
        amount_requested=amount,
        fee_open=0.001,  # Estimated 0.1% fee
        fee_open_cost=value_usdt * 0.001,
        fee_open_currency="USDT",
        open_rate=current_price,
        open_rate_requested=current_price,
        open_date=datetime.now(timezone.utc),
        exchange="okx",
        is_open=True,
        enter_tag="wallet_sync_import",
        strategy=config.get("strategy", "CustomBestStrategy"),
        leverage=1.0,
        orders=[],
    )

    # Simulate an order
    order = Order(
        ft_trade_id=trade.id,
        ft_order_side="buy",
        ft_pair=pair,
        ft_is_open=False,
        order_id=f"sync_{int(datetime.now().timestamp())}_{coin}",
        status="closed",
        symbol=pair,
        order_type="market",
        side="buy",
        price=current_price,
        average=current_price,
        amount=amount,
        filled=amount,
        remaining=0.0,
        cost=value_usdt,
        order_filled_date=datetime.now(timezone.utc),
        order_filled_price=current_price,
    )
    trade.orders.append(order)

    Trade.session.add(trade)
    Trade.session.add(order)
    Trade.commit()

    print(f"-> ➕ Successfully imported {pair} into Freqtrade DB.")

print("\n=== Scanning DB for Missing Wallet Balances ===")
# Now go the other way: if the DB has an open trade but the wallet says we have 0, delete it from DB.
for trade in open_trades_query:
    coin = trade.pair.split("/")[0]

    # Check what the wallet says
    wallet_info = balances.get(coin, {})
    if isinstance(wallet_info, dict):
        amount = wallet_info.get("total", 0)
    else:
        amount = 0

    # If the wallet has less than 1% of what the DB thinks we have, it was probably manually sold.
    # (Checking against a small threshold instead of exactly 0 to handle dust leftovers)
    if amount < (trade.amount * 0.01):
        print(
            f"🗑️ {trade.pair}: Trade ID {trade.id} is open in DB but missing from wallet (DB amt: {trade.amount:.4f}, Wallet amt: {amount:.4f}). Removing from DB..."
        )
        Trade.session.delete(trade)
        Trade.commit()

print("\n=== Done ===")
