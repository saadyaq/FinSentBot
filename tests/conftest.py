import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    symbols = ["AAA", "AAA", "AAA", "BBB", "BBB", "CCC", "CCC", "DDD", "EEE", "FFF"]
    timestamps = pd.date_range("2025-01-01", periods=len(symbols), freq="H")
    sentiment = rng.uniform(-0.5, 0.9, size=len(symbols))
    price_now = rng.uniform(50, 200, size=len(symbols))
    variation = rng.uniform(-0.05, 0.05, size=len(symbols))
    price_future = price_now * (1 + variation)
    actions_cycle = ["BUY", "HOLD", "SELL"]

    df = pd.DataFrame(
        {
            "symbol": symbols,
            "sentiment_score": sentiment,
            "price_now": price_now,
            "price_future": price_future,
            "variation": variation,
            "action": [actions_cycle[i % 3] for i in range(len(symbols))],
            "news_timestamp": timestamps,
            "price_timestamp": timestamps + pd.Timedelta(minutes=5),
        }
    )
    return df


@pytest.fixture
def sample_csv(tmp_path, sample_dataframe) -> str:
    csv_path = tmp_path / "sample_training.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return str(csv_path)
