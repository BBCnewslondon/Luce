import os
from dotenv import load_dotenv

# Ensure we load environment variables
load_dotenv()

from data_ingestion.oanda_client import OandaClient

# The 7 Major Currency Pairs
MAJOR_PAIRS = [
    "EUR_USD",
    "USD_JPY",
    "GBP_USD",
    "USD_CHF",
    "AUD_USD",
    "USD_CAD",
    "NZD_USD",
]


def main() -> None:
    print("========================================")
    print("     OANDA Account & Majors Checker     ")
    print("========================================")

    print("\n1. Initializing OANDA Client...")
    try:
        client = OandaClient()
    except Exception as e:
        print(f"  [FAIL] Failed to initialize client: {e}")
        return

    print(f"  [OK] Client initialized.")
    print(f"  Account ID:   {client.config.account_id}")
    print(f"  Environment:  {client.config.environment}")

    print("\n2. Testing get_open_positions()...")
    try:
        positions = client.get_open_positions()
        print(f"  [OK] Open positions: {len(positions)}")
        if not positions.empty:
            for row in positions.itertuples(index=False):
                print(f"    - {row.symbol}: {row.net_units} units")
    except Exception as e:
        print(f"  [FAIL] Could not retrieve positions: {e}")

    print(f"\n3. Querying authorized major currency pairs ({len(MAJOR_PAIRS)} pairs)...")
    try:
        # Request only the major pairs from OANDA
        instruments_data = client.list_account_instruments(
            instruments_filter=MAJOR_PAIRS,
            only_tradeable=True,
        )
        available_names = {row["name"]: row for row in instruments_data}

        print(f"  Found {len(available_names)} / {len(MAJOR_PAIRS)} major pairs on this account:\n")
        print(f"  {'Pair':<10} {'Display':<10} {'Margin Rate':<14} {'Pip Loc':<10} {'Status'}")
        print(f"  {'-'*10} {'-'*10} {'-'*14} {'-'*10} {'-'*10}")

        for pair in MAJOR_PAIRS:
            info = available_names.get(pair)
            if info:
                display = info.get("displayName", pair)
                margin = f"{float(info.get('marginRate', 0)):.2%}"
                pip_loc = str(info.get("pipLocation", "N/A"))
                print(f"  {pair:<10} {display:<10} {margin:<14} {pip_loc:<10} [OK] Available")
            else:
                print(f"  {pair:<10} {'-':<10} {'-':<14} {'-':<10} [MISSING]")

    except Exception as e:
        print(f"  [FAIL] Failed to fetch instruments: {e}")
        return

    print("\n4. Verifying candle data feed for major pairs (M5)...")
    for pair in MAJOR_PAIRS:
        try:
            df = client.get_candles(symbol=pair, granularity="M5", count=5)
            if not df.empty:
                last_row = df.iloc[-1]
                last_time = last_row["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC")
                last_close = last_row["close"]
                spread = f"{last_row['spread']:.5f}" if last_row.get("spread") is not None else "N/A"
                print(f"  [OK] {pair:<8} Latest M5: {last_close:.5f} | Spread: {spread} ({last_time})")
            else:
                print(f"  [WARN] {pair:<8} Returned empty candles DataFrame")
        except Exception as e:
            print(f"  [FAIL] {pair:<8} Candle fetch error: {e}")

    print("\n========================================")
    print("              Check Complete            ")
    print("========================================")


if __name__ == "__main__":
    main()
