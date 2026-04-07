# News-Vol-Forecast

Fusing news headlines and market data for multimodal stock volatility forecasting.

## Base Dataset

- **Source**: Kaggle — news headlines paired with OHLCV windows
- **Structure**: each row = one headline + 10-day pre-event OHLCV + 1-day post-event
- **Raw size**: ~850k rows across ~3,580 unique tickers
- **Link**: [https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests)

## Initial Data Curation and Cleaning

| Step | Output | Rows |
| ---- | ------ | ---- |
| Filter to EQUITY only (drop ETF, MutualFund) | `ticker_quotetype.csv` | 653,926 |
| Join sector & industry from yfinance | `ticker_metadata.csv` | N/A |
| Drop tickers with missing metadata (14 tickers) | `expanded_equities.csv/parquet` | 650,985 |
| Deduplicate to stock-day level (intermediate, not saved) | — | 381,047 |
| Fetch earnings dates from yfinance (1,819 tickers, 129k records) | `earnings_dates.csv` | 129,627 |
| Inner join stock-day with earnings (33.5% match rate) | `earnings_ONLY_joint.parquet` | 43,268 |

## Key Files

```text
project/
├── helper.py                            # all fetch, EDA, and utility functions
├── data_curation.ipynb                  # step-by-step curation notebook
└── EDA/
    ├── original_data/                   # raw Kaggle CSVs
    └── data/
        ├── expanded_equities.parquet    # news-centric equity dataset (650k rows x 61 cols)
        ├── earnings_ONLY_joint.parquet  # OHLCV rows matched to earnings events (43k rows x 62 cols)
        ├── ticker_quotetype.csv         # quote type per ticker
        ├── ticker_metadata.csv          # sector & industry per ticker
        └── earnings_dates.csv           # earnings dates with EPS estimate/actual/surprise
```

## Environment

- Python 3.12, managed with [uv](https://github.com/astral-sh/uv)
- Notebooks currently run on Google Colab with Google Drive persistence

```bash
uv sync
```
