import json
from dotenv import load_dotenv
load_dotenv()
from data_ingestion.oanda_client import OandaClient
client = OandaClient()
raw = client.list_account_instruments(only_tradeable=False)
for r in raw:
    if "EUR_USD" in r.get("name", ""):
        print(json.dumps(r, indent=2))
