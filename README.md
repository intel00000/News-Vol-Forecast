# News-Vol-Forecast

Fusing news headlines and market data for multimodal stock volatility forecasting.

## Base Dataset

- **Source**: Kaggle — news headlines paired with OHLCV windows
- **Structure**: each row = one headline + 10-day pre-event OHLCV + 1-day post-event
- **Raw size**: ~850k rows across ~3,580 uniquetickers
- **Link**: [https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests)

## Initial Data Curation and Cleaning

| Step | Output | Rows |
|------|--------|------|
| Filter to EQUITY only (drop ETF, MutualFund) | `ticker_quotetype.csv` | 653,926 |
| Join sector & industry from yfinance | `ticker_metadata.csv` | — |
| Drop tickers with missing metadata (14 tickers) | `expanded_equities.parquet` | 650,985 |
| Earnings dates with BMO/AMC timestamps | `earnings_dates.csv` | **TBD, challenging data source** |

## Key Files

```
project/
├── helper.py                          # all fetch, EDA, and utility functions
├── data_curation.ipynb                # step-by-step curation notebook
└── EDA/
    ├── original_data/                 # raw Kaggle CSVs
    └── data/
        ├── expanded_equities.parquet  # cleaned equity dataset (650k rows x 61 cols)
        ├── ticker_quotetype.csv       # quote type per ticker
        ├── ticker_metadata.csv        # sector & industry per ticker
        └── earnings_dates.csv         # earnings dates with BMO/AMC timestamps
```

## Environment

- Python 3.12, managed with [uv](https://github.com/astral-sh/uv)
- Notebooks currently run on Google Colab with Google Drive persistence

```bash
uv sync
```
