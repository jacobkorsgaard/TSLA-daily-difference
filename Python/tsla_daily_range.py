"""Replicate the core TSLA daily trading-range analysis in Python."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "Results" / "Graphs" / "Python"
TRAIN_END = "2026-01-12"
TEST_START = "2026-01-13"
TEST_END = "2026-01-21"


def download_daily_range() -> pd.Series:
    """Download unadjusted TSLA prices and return High minus Low."""
    prices = yf.download(
        "TSLA",
        start="2024-10-01",
        # Match the R download: Yahoo treats 2026-01-22 as an exclusive bound.
        end="2026-01-22",
        auto_adjust=False,
        progress=False,
    )
    if prices.empty:
        raise RuntimeError("No TSLA observations were downloaded.")

    daily_range = prices["High"] - prices["Low"]
    if isinstance(daily_range, pd.DataFrame):
        daily_range = daily_range.iloc[:, 0]
    return daily_range.rename("TSLA daily range").dropna()


def save_series_diagnostics(series: pd.Series, filename: str, title: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 8))
    axes[0].plot(series.index, series.values)
    axes[0].set_title(title)
    plot_acf(series, ax=axes[1], lags=30, zero=False)
    plot_pacf(series, ax=axes[2], lags=30, zero=False, method="ywm")
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=150)
    plt.close(fig)


def save_residual_diagnostics(residuals: pd.Series, filename: str, title: str) -> None:
    clean = pd.Series(residuals).dropna()
    fig, axes = plt.subplots(3, 1, figsize=(9, 8))
    axes[0].plot(clean.values)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title(title)
    plot_acf(clean, ax=axes[1], lags=20, zero=False)
    plot_pacf(clean, ax=axes[2], lags=20, zero=False, method="ywm")
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=150)
    plt.close(fig)


def forecast_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    errors = actual.to_numpy() - np.asarray(predicted)
    return {
        "ME": float(np.mean(errors)),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "MAE": float(np.mean(np.abs(errors))),
    }


def save_forecast(model, train: pd.Series, test: pd.Series, filename: str, title: str):
    prediction = model.get_forecast(steps=len(test))
    mean = np.asarray(prediction.predicted_mean)
    interval = np.asarray(prediction.conf_int(alpha=0.05))

    fig, ax = plt.subplots(figsize=(9, 5))
    recent = train.iloc[-30:]
    ax.plot(recent.index, recent.values, label="Training data")
    ax.plot(test.index, test.values, "o-", label="Observed test data")
    ax.plot(test.index, mean, label="Forecast")
    ax.fill_between(test.index, interval[:, 0], interval[:, 1], alpha=0.2)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=150)
    plt.close(fig)
    return mean


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    daily_range = download_daily_range()
    train = daily_range.loc[:TRAIN_END]
    test = daily_range.loc[TEST_START:TEST_END]

    if len(test) != 6:
        raise RuntimeError(f"Expected 6 test observations, found {len(test)}.")

    save_series_diagnostics(train, "raw_data.png", "TSLA Daily Trading Range")

    # An intercept is included because the series has a clearly nonzero mean.
    adf_stat, adf_pvalue, used_lags, *_ = adfuller(
        train, regression="c", autolag="BIC", result_object=False
    )
    print(f"ADF: statistic={adf_stat:.4f}, p-value={adf_pvalue:.4f}, lags={used_lags}")

    models = {
        "AR(1)": ARIMA(train.to_numpy(), order=(1, 0, 0), trend="c").fit(),
        "AR(2)": ARIMA(train.to_numpy(), order=(2, 0, 0), trend="c").fit(),
        "ARMA(1,1)": ARIMA(train.to_numpy(), order=(1, 0, 1), trend="c").fit(),
    }

    for name, model in models.items():
        slug = name.lower().replace("(", "_").replace(",", "_").replace(")", "")
        save_residual_diagnostics(model.resid, f"{slug}_diagnostics.png", f"{name} residuals")
        residuals = pd.Series(model.resid).dropna()
        ljung_box = acorr_ljungbox(residuals, lags=[10], return_df=True)
        squared_lb = acorr_ljungbox(residuals**2, lags=[10], return_df=True)
        print(
            f"{name}: AIC={model.aic:.2f}, BIC={model.bic:.2f}, "
            f"Ljung-Box p(10)={ljung_box['lb_pvalue'].iloc[0]:.4f}, "
            f"squared-residual p(10)={squared_lb['lb_pvalue'].iloc[0]:.4f}"
        )

    for name in ("AR(2)", "ARMA(1,1)"):
        filename = name.lower().replace("(", "_").replace(",", "_").replace(")", "")
        mean = save_forecast(
            models[name], train, test, f"{filename}_forecast.png", f"{name} forecast"
        )
        print(f"{name} forecast accuracy: {forecast_metrics(test, mean)}")


if __name__ == "__main__":
    main()
