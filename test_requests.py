import os
import requests
from dotenv import load_dotenv
load_dotenv()

token = os.getenv("OANDA_API_TOKEN")
account = os.getenv("OANDA_ACCOUNT_ID")

headers = {
    "Authorization": f"Bearer {token}",
    "Accept-Datetime-Format": "RFC3339"
}

# Test 1: get_open_positions
url1 = f"https://api-fxpractice.oanda.com/v3/accounts/{account}/openPositions"
r1 = requests.get(url1, headers=headers)
print("openPositions:", r1.status_code, r1.text)

# Test 2: candles
url2 = "https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles?granularity=M5&count=10"
r2 = requests.get(url2, headers=headers)
print("candles:", r2.status_code, r2.text)
