"""
Commitment of Traders (COT) data fetcher.

Fetches positioning data from CFTC to gauge market sentiment.
Data is released weekly (Friday) with a 3-day lag to actual positions.
"""

import io
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
import requests
from loguru import logger


@dataclass
class CotConfig:
    """Configuration for COT data fetcher."""

    report_type: str = "disaggregated"  # "legacy" or "disaggregated"
    cache_dir: Optional[str] = None


class CotFetcher:
    """
    Fetches CFTC Commitment of Traders report data.

    COT data shows positioning of different trader types (commercial,
    non-commercial, etc.) and can signal potential reversals when
    positioning reaches extremes.

    CRITICAL: COT data has a built-in 3-day lag. Reports released on
    Friday reflect positions as of the previous Tuesday. This must be
    accounted for to avoid look-ahead bias.
    """

    # CFTC data URLs
    CFTC_BASE_URL = "https://www.cftc.gov/files/dea/history"

    # Currency futures contract codes (CME)
    CURRENCY_CONTRACTS = {
        "EUR_USD": "099741",  # Euro FX
        "GBP_USD": "096742",  # British Pound
        "USD_JPY": "097741",  # Japanese Yen (inverted for USD/JPY)
        "USD_CHF": "092741",  # Swiss Franc (inverted)
        "AUD_USD": "232741",  # Australian Dollar
        "USD_CAD": "090741",  # Canadian Dollar (inverted)
    }

    # 3-day publication lag (Tuesday positions, Friday release)
    PUBLICATION_LAG_DAYS = 3

    def __init__(self, config: Optional[CotConfig] = None):
        """
        Initialize COT fetcher.

        Args:
            config: CotConfig instance.
        """
        self.config = config or CotConfig()
        self._cache: Dict[int, pd.DataFrame] = {}
        logger.info(f"COT fetcher initialized ({self.config.report_type} report)")

    def fetch(
        self,
        year: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch COT data for specified period.

        Args:
            year: Specific year to fetch (fetches full year).
            start_date: Start date for filtering.
            end_date: End date for filtering.

        Returns:
            DataFrame with COT positioning data.
        """
        year = year or datetime.now().year

        if year in self._cache:
            df = self._cache[year]
        else:
            df = self._fetch_year_data(year)
            if not df.empty:
                self._cache[year] = df

        if df.empty:
            return df

        # Filter by date range if specified
        if start_date:
            df = df[df["report_date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["report_date"] <= pd.Timestamp(end_date)]

        return df.reset_index(drop=True)

    def _fetch_year_data(self, year: int) -> pd.DataFrame:
        """
        Fetch COT data for a specific year from CFTC.

        Args:
            year: Year to fetch.

        Returns:
            Raw COT DataFrame.
        """
        # Construct URL based on report type
        if self.config.report_type == "disaggregated":
            filename = f"fut_disagg_txt_{year}.zip"
        else:
            filename = f"fut_fin_txt_{year}.zip"

        url = f"{self.CFTC_BASE_URL}/{filename}"
        logger.info(f"Fetching COT data from {url}")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch COT data: {e}")
            return pd.DataFrame()

        return self._parse_cot_zip(response.content, year)

    def _parse_cot_zip(self, zip_content: bytes, year: int) -> pd.DataFrame:
        """
        Parse COT data from CFTC zip file.

        Args:
            zip_content: Raw zip file bytes.
            year: Year of the data.

        Returns:
            Parsed DataFrame with standardized columns.
        """
        try:
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                # Find the txt file in the zip
                txt_files = [f for f in zf.namelist() if f.endswith(".txt")]
                if not txt_files:
                    logger.error("No .txt file found in COT zip")
                    return pd.DataFrame()

                with zf.open(txt_files[0]) as f:
                    df = pd.read_csv(f)
        except Exception as e:
            logger.error(f"Failed to parse COT zip: {e}")
            return pd.DataFrame()

        return self._normalize_cot_data(df)

    def _normalize_cot_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize COT data to standard format.

        Extracts relevant columns and calculates net positioning
        for currency futures.
        """
        # Standardize column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Filter to currency contracts only
        if "cftc_contract_market_code" in df.columns:
            code_col = "cftc_contract_market_code"
        elif "cftc_commodity_code" in df.columns:
            code_col = "cftc_commodity_code"
        else:
            logger.warning("Cannot identify contract code column")
            return pd.DataFrame()

        currency_codes = list(self.CURRENCY_CONTRACTS.values())
        df = df[df[code_col].astype(str).isin(currency_codes)]

        if df.empty:
            return df

        # Extract key positioning columns
        records = []
        for _, row in df.iterrows():
            contract_code = str(row[code_col])

            # Map contract code to symbol
            symbol = None
            for sym, code in self.CURRENCY_CONTRACTS.items():
                if code == contract_code:
                    symbol = sym
                    break

            if not symbol:
                continue

            # Get report date
            if "report_date_as_yyyy-mm-dd" in row:
                report_date = pd.to_datetime(row["report_date_as_yyyy-mm-dd"])
            elif "as_of_date_in_form_yymmdd" in row:
                report_date = pd.to_datetime(str(row["as_of_date_in_form_yymmdd"]), format="%y%m%d")
            else:
                continue

            # Extract positioning (non-commercial = speculators)
            long_col = "noncomm_positions_long_all" if "noncomm_positions_long_all" in row else "noncommercial_positions_long_all"
            short_col = "noncomm_positions_short_all" if "noncomm_positions_short_all" in row else "noncommercial_positions_short_all"

            noncomm_long = float(row.get(long_col, row.get("noncommercial_long_all", 0)))
            noncomm_short = float(row.get(short_col, row.get("noncommercial_short_all", 0)))

            # Commercial (hedgers)
            comm_long = float(row.get("comm_positions_long_all", row.get("commercial_long_all", 0)))
            comm_short = float(row.get("comm_positions_short_all", row.get("commercial_short_all", 0)))

            # Open interest
            oi = float(row.get("open_interest_all", row.get("oi_all", 0)))

            records.append({
                "report_date": report_date,
                "symbol": symbol,
                "noncomm_net": noncomm_long - noncomm_short,
                "noncomm_long": noncomm_long,
                "noncomm_short": noncomm_short,
                "comm_net": comm_long - comm_short,
                "open_interest": oi,
            })

        result = pd.DataFrame(records)
        if not result.empty:
            result = result.sort_values(["symbol", "report_date"]).reset_index(drop=True)

        logger.info(f"Parsed {len(result)} COT records")
        return result

    def get_symbol_positioning(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Get COT positioning for a specific currency pair.

        Args:
            symbol: Currency pair (e.g., "EUR_USD").
            start_date: Start date.
            end_date: End date.

        Returns:
            DataFrame with positioning data for the symbol.
        """
        if symbol not in self.CURRENCY_CONTRACTS:
            logger.warning(f"No COT data available for {symbol}")
            return pd.DataFrame()

        # Determine years to fetch
        start_year = (start_date or datetime(2010, 1, 1)).year
        end_year = (end_date or datetime.now()).year

        all_data = []
        for year in range(start_year, end_year + 1):
            df = self.fetch(year=year)
            if not df.empty:
                all_data.append(df)

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)
        combined = combined[combined["symbol"] == symbol]

        if start_date:
            combined = combined[combined["report_date"] >= pd.Timestamp(start_date)]
        if end_date:
            combined = combined[combined["report_date"] <= pd.Timestamp(end_date)]

        return combined.drop_duplicates(subset=["report_date"]).sort_values("report_date").reset_index(drop=True)

    def forward_fill_to_hourly(
        self,
        cot_df: pd.DataFrame,
        target_timestamps: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Forward-fill weekly COT data to hourly timestamps with proper lag.

        CRITICAL: Applies 3-day publication lag to prevent look-ahead bias.
        COT report dated Tuesday is not available until Friday evening.

        Args:
            cot_df: COT DataFrame from fetch() or get_symbol_positioning().
            target_timestamps: Hourly timestamps to align to.

        Returns:
            COT data aligned to target timestamps with lag applied.
        """
        if cot_df.empty:
            return pd.DataFrame({
                "timestamp": target_timestamps,
                "cot_net_long": [None] * len(target_timestamps),
                "cot_change": [None] * len(target_timestamps),
            })

        # Apply publication lag: data becomes available 3 days after report date
        cot_df = cot_df.copy()
        cot_df["available_date"] = cot_df["report_date"] + timedelta(days=self.PUBLICATION_LAG_DAYS)

        # Reindex to available date (when data would actually be known)
        cot_hourly = cot_df.set_index("available_date")[["noncomm_net", "open_interest"]]
        cot_hourly = cot_hourly.rename(columns={"noncomm_net": "cot_net_long"})

        # Reindex to target timestamps
        target_df = pd.DataFrame(index=target_timestamps)
        merged = target_df.join(cot_hourly, how="left")

        # Forward fill (each hour gets last known COT value)
        merged["cot_net_long"] = merged["cot_net_long"].ffill()
        merged["open_interest"] = merged["open_interest"].ffill()

        # Calculate change in positioning (week-over-week)
        merged["cot_change"] = merged["cot_net_long"].pct_change(periods=168)  # 168 hours = 1 week

        # Normalize by open interest for comparability
        merged["cot_net_pct"] = merged["cot_net_long"] / merged["open_interest"].replace(0, 1)

        result = merged.reset_index().rename(columns={"index": "timestamp"})
        return result

    def calculate_positioning_extremes(
        self,
        cot_df: pd.DataFrame,
        lookback_weeks: int = 52,
    ) -> pd.DataFrame:
        """
        Calculate percentile ranks of positioning over lookback period.

        Useful for identifying crowded trades and potential reversals.

        Args:
            cot_df: COT positioning DataFrame.
            lookback_weeks: Number of weeks for percentile calculation.

        Returns:
            DataFrame with percentile rank columns added.
        """
        df = cot_df.copy()

        # Rolling percentile rank (0 = most short, 1 = most long)
        df["cot_percentile"] = df["cot_net_long"].rolling(
            window=lookback_weeks
        ).apply(lambda x: (x.rank().iloc[-1] - 1) / (len(x) - 1) if len(x) > 1 else 0.5)

        return df
