# Tesla Daily Trading Range

A small time-series project comparing models for Tesla's daily trading range—the daily high minus the daily low—from October 2024 to January 2026.

The analysis compares AR(1), AR(2), and ARMA(1,1) specifications using model diagnostics, information criteria, and a six-observation out-of-sample test set. The AR(2) and ARMA(1,1) models perform similarly; the report ultimately favors AR(2) for its interpretability and slightly lower in-sample information criteria.

## Repository contents

- `R/`: original R analysis used for the report
- `Python/`: a Python implementation of the main analysis
- `Results/TSLA Daily Trading Range - Report.pdf`: public report with personal student ID removed
- `Results/Graphs/`: figures from the R and Python analyses

## Run the analysis

The scripts download TSLA prices from Yahoo Finance, so an internet connection is required.

### R

Install the packages `quantmod`, `forecast`, `urca`, and `TSA`, then run the script in `R/`.

### Python

```bash
python3 -m venv Python/.venv
source Python/.venv/bin/activate
pip install -r Python/requirements.txt
python Python/tsla_daily_range.py
```
