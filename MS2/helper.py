"""
helpers.py — stock market prediction project
"""

import time as _time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────
# ROOT must be set by the caller before using any path constants.
# Local:  ROOT = Path(".")
# Colab:  ROOT = Path("/content/drive/MyDrive/cs209b/project")
# Override in the notebook with:  import helper; helper.ROOT = Path("...")
ROOT: Path = Path(".")
ORIG_DIR: Path = ROOT / "EDA" / "original_data"
DATA_DIR: Path = ROOT / "EDA" / "data"
OHLCV_DIR: Path = DATA_DIR / "local_ohlcv_10y_full"


def set_root(path: str | Path):
    """
    Call this at the top of every notebook to point all path constants at the
    correct project root for the current environment. Returns the four path
    constants so the notebook can rebind its local names.

    Usage
    -----
    Local:
        ROOT, ORIG_DIR, DATA_DIR, OHLCV_DIR = set_root(".")
    Colab (Google Drive mounted):
        ROOT, ORIG_DIR, DATA_DIR, OHLCV_DIR = set_root("/content/drive/MyDrive/cs209b/project")
    """
    global ROOT, ORIG_DIR, DATA_DIR, OHLCV_DIR
    ROOT = Path(path)
    ORIG_DIR = ROOT / "EDA" / "original_data"
    DATA_DIR = ROOT / "EDA" / "data"
    OHLCV_DIR = DATA_DIR / "local_ohlcv_10y_full"
    print(f"ROOT      : {ROOT.resolve()}")
    print(f"ORIG_DIR  : {ORIG_DIR}")
    print(f"DATA_DIR  : {DATA_DIR}")
    print(f"OHLCV_DIR : {OHLCV_DIR}")
    return ROOT, ORIG_DIR, DATA_DIR, OHLCV_DIR


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOADERS
# ═══════════════════════════════════════════════════════════════════════════


def load_news_csv(path: str | Path) -> pd.DataFrame:
    """
    Generic loader for any news CSV (analyst ratings or partner headlines).
    Parses the date column to tz-aware UTC, drops the unnamed index column,
    and filters out rows before 2005 to remove bad epoch entries.
    """
    df = pd.read_csv(path, index_col=0)
    df["date"] = pd.to_datetime(df["date"], format="mixed", utc=True)
    return df[df["date"].dt.year >= 2005].reset_index(drop=True)


def load_expanded(path: str | Path) -> pd.DataFrame:
    """
    Load the expanded dataset (news + OHLCV windows).
    Each row: headline, stock, date, t-10…t-1 OHLCV, t+1 OHLCV.
    """
    df = pd.read_csv(path, index_col=0)
    df["date"] = pd.to_datetime(df["date"], format="mixed", utc=True)
    return df


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    """Load a single per-ticker OHLCV CSV. Columns: Date, Open, High, Low, Close, Volume."""
    return pd.read_csv(path, parse_dates=["Date"])


# ═══════════════════════════════════════════════════════════════════════════
# 2. EDA — EXISTING DATA
# ═══════════════════════════════════════════════════════════════════════════


def eda_expanded_overview(df):
    print(f"Shape         : {df.shape}")
    print(f"Date range    : {df['date'].min().date()}  →  {df['date'].max().date()}")
    print(f"Unique tickers: {df['stock'].nunique():,}")
    print()
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    print("Columns with nulls:")
    print(nulls.to_string() if len(nulls) else "  none")


def plot_yearly_counts(df, date_col="date", title="Rows per Year"):
    yearly = pd.to_datetime(df[date_col], utc=True).dt.year.value_counts().sort_index()
    ax = yearly.plot(kind="bar", figsize=(11, 4))
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("# Rows")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height()):,}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    plt.show()


def plot_ticker_coverage(df, stock_col="stock", top_n=30):
    counts = df[stock_col].value_counts()
    cum_pct = counts.cumsum() / counts.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    counts.head(top_n).plot(kind="bar", ax=axes[0])
    axes[0].set_title(f"Top {top_n} Tickers by Row Count")
    axes[0].set_ylabel("# Rows")
    axes[0].tick_params(axis="x", rotation=90)

    axes[1].plot(range(1, len(cum_pct) + 1), cum_pct.values)
    for pct in [50, 80, 95]:
        n = int((cum_pct <= pct).sum())
        axes[1].axhline(pct, color="gray", linestyle="--", linewidth=0.8)
        axes[1].text(n + 20, pct + 0.5, f"{pct}% @ {n} tickers", fontsize=8)
    axes[1].set_title("Cumulative Coverage vs # Tickers")
    axes[1].set_xlabel("# Tickers (ranked)")
    axes[1].set_ylabel("Cumulative % of rows")
    plt.tight_layout()
    plt.show()

    for thresh in [100, 250, 500, 1000]:
        n = (counts >= thresh).sum()
        pct = counts[counts >= thresh].sum() / counts.sum() * 100
        print(f"  ≥{thresh:>4} rows : {n:>4} tickers  ({pct:.1f}% of all rows)")


def plot_ohlcv_window(df, n_sample=200):
    close_cols = [f"t_minus_{i}_close" for i in range(10, 0, -1)] + ["t_plus_1_close"]
    x_labels = list(range(-10, 0)) + [1]
    sample = df[close_cols].dropna().head(n_sample)
    norm = sample.div(sample["t_minus_1_close"], axis=0)

    _, ax = plt.subplots(figsize=(12, 4))
    for _, row in norm.iterrows():
        ax.plot(x_labels, row.values, alpha=0.05, color="steelblue")
    ax.plot(x_labels, norm.mean().values, color="red", linewidth=2, label="mean")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="t=0 (no data)")
    ax.axvline(1, color="green", linestyle="--", linewidth=1, label="t+1")
    ax.set_xticks(x_labels)
    ax.set_xlabel("Trading days relative to news")
    ax.set_ylabel("Normalized close  (t−1 = 1.0)")
    ax.set_title(f"OHLCV Window — {n_sample} sample rows")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_next_day_return(df):
    ret = (df["t_plus_1_close"] - df["t_minus_1_close"]) / df["t_minus_1_close"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    ret.clip(-0.15, 0.15).hist(bins=120, ax=axes[0], edgecolor="none")
    axes[0].axvline(0, color="red", linewidth=1)
    axes[0].set_title("Next-Day Return Distribution  (clipped ±15%)")
    axes[0].set_xlabel("Return")
    ret.groupby(df["date"].dt.year).mean().plot(kind="bar", ax=axes[1])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Mean Next-Day Return by Year")
    axes[1].set_xlabel("Year")
    plt.tight_layout()
    plt.show()
    print(ret.describe().round(4))
    return ret


def eda_raw_news(ar, ph, expanded):
    print("=== raw_analyst_ratings ===")
    print(
        f"  rows: {len(ar):,}  |  tickers: {ar['stock'].nunique():,}  |  publishers: {ar['publisher'].nunique():,}"
    )
    print(f"  date: {ar['date'].min().date()}  →  {ar['date'].max().date()}")
    print()
    print("=== raw_partner_headlines ===")
    print(
        f"  rows: {len(ph):,}  |  tickers: {ph['stock'].nunique():,}  |  publishers: {ph['publisher'].nunique():,}"
    )
    print(f"  date: {ph['date'].min().date()}  →  {ph['date'].max().date()}")
    print()
    dropped = len(ar) - len(expanded)
    print(f"Dropped (no local OHLCV): {dropped:,}  ({dropped / len(ar) * 100:.1f}%)")

    yr_ar = ar["date"].dt.year.value_counts().sort_index().rename("raw_analyst")
    yr_exp = expanded["date"].dt.year.value_counts().sort_index().rename("expanded")
    yr_ph = ph["date"].dt.year.value_counts().sort_index().rename("partner")
    yearly = pd.concat([yr_ar, yr_exp, yr_ph], axis=1).fillna(0).astype(int)
    ax = yearly.plot(kind="bar", figsize=(12, 4))
    ax.set_title("Headlines per Year: Raw vs Expanded vs Partner")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    plt.show()
    print(yearly.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    ar["publisher"].value_counts().head(15).plot(kind="barh", ax=axes[0])
    axes[0].set_title("Analyst Ratings — Top 15 Publishers")
    axes[0].invert_yaxis()
    ph["publisher"].value_counts().head(15).plot(kind="barh", ax=axes[1])
    axes[1].set_title("Partner Headlines — Top 15 Publishers")
    axes[1].invert_yaxis()
    plt.tight_layout()
    plt.show()


def eda_ohlcv_files(ar):
    files = sorted(OHLCV_DIR.glob("*.csv"))
    available = {f.stem for f in files}
    aapl = pd.read_csv(OHLCV_DIR / "AAPL.csv")
    print(f"Total OHLCV files : {len(files):,}")
    print(f"Columns           : {list(aapl.columns)}")
    print(f"Rows per ticker   : {len(aapl)}  (e.g. AAPL)")
    print(f"Date range        : {aapl['Date'].min()}  →  {aapl['Date'].max()}")

    news_tickers = set(ar["stock"].unique())
    without_ohlcv = news_tickers - available
    miss_counts = ar[ar["stock"].isin(without_ohlcv)]["stock"].value_counts()
    print()
    print(f"With OHLCV    : {len(news_tickers & available):,}")
    print(
        f"Without OHLCV : {len(without_ohlcv):,}  →  {miss_counts.sum():,} headlines ({miss_counts.sum() / len(ar) * 100:.1f}%)"
    )
    print("Top 10 missing:", miss_counts.head(10).to_dict())

    aapl = pd.read_csv(OHLCV_DIR / "AAPL.csv", parse_dates=["Date"])
    fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
    axes[0].plot(aapl["Date"], aapl["Close"])
    axes[0].set_title("AAPL — Close 2010–2020")
    axes[0].set_ylabel("Price ($)")
    axes[1].bar(aapl["Date"], aapl["Volume"], width=1, alpha=0.5)
    axes[1].set_ylabel("Volume")
    axes[1].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.0f}M")
    )
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# 3. YFINANCE EXPLORATION
# ═══════════════════════════════════════════════════════════════════════════


def yf_probe_earnings_dates(symbol, limit=60):
    raw = yf.Ticker(symbol).get_earnings_dates(limit=limit)
    if raw is None or raw.empty:
        print(f"{symbol}: no data")
        return pd.DataFrame()
    raw.index = pd.to_datetime(raw.index, utc=True)
    raw.columns = ["eps_estimate", "reported_eps", "surprise_pct"]
    raw["hour_et"] = raw.index.tz_convert("America/New_York").hour
    raw["timing"] = raw["hour_et"].apply(lambda h: "BMO" if h < 16 else "AMC")
    raw["in_window"] = (raw.index.year >= 2010) & (raw.index.year <= 2020)
    print(
        f"{symbol}  total={len(raw)}  in_2010-2020={raw['in_window'].sum()}  {raw['timing'].value_counts().to_dict()}"
    )
    return raw.sort_index()


def plot_earnings_dates(ed, symbol):
    window = ed[ed["in_window"]].dropna(subset=["reported_eps"])
    if window.empty:
        print("No data in 2010-2020")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    window["surprise_pct"].clip(-30, 30).hist(bins=40, ax=axes[0], edgecolor="none")
    axes[0].axvline(0, color="red", linewidth=1)
    axes[0].set_title(f"{symbol} — EPS Surprise % (2010-2020)")
    window.groupby(window.index.year)["surprise_pct"].mean().plot(
        kind="bar", ax=axes[1]
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title(f"{symbol} — Mean EPS Surprise by Year")
    plt.tight_layout()
    plt.show()


def yf_probe_upgrades(symbol):
    raw = yf.Ticker(symbol).upgrades_downgrades
    if raw is None or raw.empty:
        print(f"{symbol}: no data")
        return pd.DataFrame()
    # index is tz-aware; strip timezone with tz_convert(None)
    raw.index = pd.to_datetime(raw.index, utc=True).tz_convert(None)
    raw = raw.sort_index()
    raw["sentiment"] = (
        raw["Action"]
        .map({"up": 1, "down": -1, "main": 0, "reit": 0, "init": 0})
        .fillna(0)
    )
    raw["in_window"] = (raw.index.year >= 2010) & (raw.index.year <= 2020)
    window = raw[raw["in_window"]]
    print(
        f"{symbol}  total={len(raw)}  in_2010-2020={len(window)}  range={raw.index.min().date()}→{raw.index.max().date()}"
    )
    print(window["Action"].value_counts().to_string())
    return raw


def plot_upgrades(ud, symbol):
    window = ud[ud["in_window"]]
    if window.empty:
        print("No data in 2010-2020")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    window["Action"].value_counts().plot(kind="bar", ax=axes[0])
    axes[0].set_title(f"{symbol} — Analyst Actions (2010-2020)")
    axes[0].tick_params(axis="x", rotation=0)
    window["sentiment"].resample("ME").sum().plot(ax=axes[1], color="steelblue")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title(f"{symbol} — Monthly Net Analyst Sentiment")
    plt.tight_layout()
    plt.show()


def yf_probe_info(symbols):
    keep = [
        "sector",
        "industry",
        "quoteType",
        "exchange",
        "marketCap",
        "beta",
        "trailingPE",
        "priceToBook",
        "debtToEquity",
        "returnOnEquity",
        "profitMargins",
        "shortRatio",
    ]
    rows = []
    for sym in symbols:
        info = yf.Ticker(sym).info or {}
        rows.append({"symbol": sym} | {k: info.get(k) for k in keep})
    return pd.DataFrame(rows).set_index("symbol")


def yf_coverage_test(tickers):
    rows = []
    for sym in tickers:
        try:
            ed = yf.Ticker(sym).get_earnings_dates(limit=60)
            idx = pd.to_datetime(ed.index, utc=True)
            in_win = ((idx.year >= 2010) & (idx.year <= 2020)).sum()
            hours = idx.tz_convert("America/New_York").hour
            rows.append(
                {
                    "symbol": sym,
                    "total": len(ed),
                    "in_2010_2020": in_win,
                    "bmo": int((hours < 16).sum()),
                    "amc": int((hours >= 16).sum()),
                    "ok": True,
                }
            )
        except:
            rows.append(
                {
                    "symbol": sym,
                    "total": 0,
                    "in_2010_2020": 0,
                    "bmo": 0,
                    "amc": 0,
                    "ok": False,
                }
            )
    df = pd.DataFrame(rows)
    print(
        f"Tested={len(df)}  OK={df['ok'].sum()}  Mean quarters in window={df['in_2010_2020'].mean():.1f}"
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 4. YFINANCE BATCH FETCHERS  (write output CSVs, checkpointed)
# ═══════════════════════════════════════════════════════════════════════════


def _fetch_info_batch(
    tickers: list, fields: list, out_path: str | Path, sleep: float = 0.1
) -> pd.DataFrame:
    """
    Internal helper: fetch selected info fields for a list of tickers.
    Saves to out_path after EVERY request so no progress is ever lost.
    Resumes automatically from out_path if it already exists.
    """
    out_path = Path(out_path)

    # Resume: load already-fetched rows, skip those symbols
    if out_path.exists():
        existing = pd.read_csv(out_path)
        done = set(existing["symbol"].tolist())
        rows = existing.to_dict("records")
        print(f"  Resuming: {len(done)}/{len(tickers)} already done")
    else:
        done, rows = set(), []

    remaining = [t for t in tickers if t not in done]
    total = len(tickers)

    for i, sym in enumerate(remaining):
        try:
            info = yf.Ticker(sym).info or {}
        except Exception:
            info = {}
        row: dict = {"symbol": sym}
        for f in fields:
            row[f] = info.get(f)
        rows.append(row)

        # Checkpoint after every single request
        pd.DataFrame(rows).to_csv(out_path, index=False)

        fetched = len(done) + i + 1
        if fetched % 100 == 0:
            print(f"  {fetched}/{total}")

        _time.sleep(sleep)

    result = pd.DataFrame(rows)
    print(f"  Done: {len(result)} tickers -> {out_path}")
    return result


def fetch_quote_types(
    tickers: list, out_path: str | Path, sleep: float = 0.1
) -> pd.DataFrame:
    """
    Fetch quoteType for every ticker. Saves after every request, resumes on restart.
    Returns DataFrame(symbol, quoteType).
    """
    return _fetch_info_batch(tickers, ["quoteType"], out_path, sleep=sleep)


def fetch_ticker_metadata(
    tickers: list, out_path: str | Path, sleep: float = 0.1
) -> pd.DataFrame:
    """
    Fetch sector and industry for each ticker. Saves after every request, resumes on restart.
    Returns DataFrame(symbol, sector, industry).
    """
    return _fetch_info_batch(tickers, ["sector", "industry"], out_path, sleep=sleep)


def fetch_earnings_all(
    tickers: list, out_path: str | Path, limit: int = 60, sleep: float = 0.15
) -> pd.DataFrame:
    """
    Fetch get_earnings_dates for every ticker. Saves after EVERY request.
    Resumes automatically from out_path if interrupted.

    Output columns:
        symbol, earnings_date (tz-aware UTC), eps_estimate, reported_eps, surprise_pct

    The time component of earnings_date encodes announcement timing:
        hour (ET) < 16  ->  before market open (BMO)
        hour (ET) >= 16 ->  after market close (AMC)

    A sidecar file <out_path>.attempted tracks every ticker that was tried
    (including ones that returned no data), so they are not retried on resume.
    """
    out_path = Path(out_path)
    attempted_path = out_path.with_suffix(".attempted.txt")

    # Load already-attempted tickers (tried but may have returned no data)
    if attempted_path.exists():
        done = set(attempted_path.read_text().splitlines())
    else:
        done = set()

    # Load existing earnings rows
    if out_path.exists() and out_path.stat().st_size > 0:
        existing = pd.read_csv(out_path)
        existing["earnings_date"] = pd.to_datetime(existing["earnings_date"], utc=True)
        frames = [existing]
        print(f"  Resuming: {len(done)}/{len(tickers)} tickers already attempted")
    else:
        frames = []

    remaining = [t for t in tickers if t not in done]
    total = len(tickers)

    for sym in remaining:
        try:
            raw = yf.Ticker(sym).get_earnings_dates(limit=limit)
            if raw is not None and not raw.empty:
                raw = raw.reset_index()
                raw.columns = [str(c).strip() for c in raw.columns]
                raw = raw.rename(
                    columns={
                        "Earnings Date": "earnings_date",
                        "EPS Estimate": "eps_estimate",
                        "Reported EPS": "reported_eps",
                        "Surprise(%)": "surprise_pct",
                    }
                )
                raw["symbol"] = sym
                raw["earnings_date"] = pd.to_datetime(raw["earnings_date"], utc=True)
                frames.append(
                    raw[
                        [
                            "symbol",
                            "earnings_date",
                            "eps_estimate",
                            "reported_eps",
                            "surprise_pct",
                        ]
                    ]
                )
        except Exception:
            pass

        # Mark as attempted (even if no data returned) and checkpoint
        done.add(sym)
        attempted_path.write_text("\n".join(done))
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)

        fetched = len(done)
        if fetched % 100 == 0:
            print(f"  {fetched}/{total}")

        _time.sleep(sleep)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if result.empty:
        print("  Done: no earnings records found")
    else:
        print(
            f"  Done: {len(result):,} records for {result['symbol'].nunique()} tickers -> {out_path}"
        )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 5. DATA UTILITIES
# ═══════════════════════════════════════════════════════════════════════════


def add_returns(ohlcv):
    df = ohlcv.copy()
    df["return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def add_parkinson_vol(ohlcv, window=21):
    df = ohlcv.copy()
    log_hl_sq = np.log(df["High"] / df["Low"]) ** 2
    df[f"parkinson_vol_{window}d"] = (
        log_hl_sq.rolling(window).mean() / (4 * np.log(2))
    ) ** 0.5
    return df
