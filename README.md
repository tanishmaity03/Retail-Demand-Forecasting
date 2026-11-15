# Retail Demand Forecasting: Statistical, Machine Learning and Deep Learning Models

Comparative study of five models for weekly retail sales forecasting, with a cost-based analysis of forecast error.
Author: Tanish Maity, M.Tech Data Science and Analytics, Lovely Professional University.

## Overview

Five models are benchmarked on weekly sales from the Walmart Recruiting Store Sales Forecasting dataset (45 stores, 143 weeks, 2010-02-05 to 2012-10-26):

- Seasonal Naive (baseline)
- SARIMA
- Prophet
- XGBoost (lag, rolling, calendar, holiday and promotion features)
- LSTM

The first 117 weeks are used for training and the last 26 weeks for testing (time-based split, no shuffling). Forecasts are multi-step and recursive.

## Results (26-week test set)

| Model | MAE ($) | RMSE ($) | MAPE (%) | Improvement vs baseline (%) |
|---|---|---|---|---|
| Seasonal Naive | 1,181,619 | 1,455,381 | 2.499 | 0.0 |
| SARIMA | 891,485 | 1,179,958 | 1.891 | 24.3 |
| Prophet | 990,257 | 1,338,149 | 2.095 | 16.1 |
| XGBoost | 1,034,207 | 1,373,737 | 2.183 | 12.6 |
| LSTM | 1,354,849 | 1,688,761 | 2.929 | -17.2 |

Key findings:

- SARIMA gave the lowest error and the lowest weighted error cost at every tested cost ratio (1:1, 1:2, 1:4).
- XGBoost's most important features were the 52-week lag, week of year and the 4-week lag.
- The LSTM did not beat the baseline on this small dataset.

All figures and tables are in the `results/` folder.

## How to run

1. Download `train.csv` and `features.csv` from the Kaggle competition
   [Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting) (a free Kaggle account and acceptance of the competition rules are required). The data is not included in this repository.
2. Place both files in the same folder as `retail_forecasting.py`.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run:
   ```
   python retail_forecasting.py
   ```
   Runtime is roughly 5-15 minutes (SARIMA is the slowest step). Outputs are saved to `results/`.

Settings such as the number of stores, test length and cost weights are in the CONFIG block at the top of the script. Random seeds are fixed to 42.

## Outputs

- `table2_accuracy.csv`: model comparison
- `table3_ablation.csv`: XGBoost feature ablation
- `table4_cost.csv`, `table4_sensitivity.csv`: forecast error cost and its sensitivity to the cost ratio
- `forecasts.csv`: test-period forecasts of every model
- `fig1_forecasts.png`, `fig2_xgb_importance.png`, `eda_*.png`: figures
- `summary.txt`: run log with the main numbers

## Limitations

- One dataset, one 26-week test window and one random seed; no statistical significance tests.
- The test window (May to October 2012) excludes the Thanksgiving and Christmas peaks.
- Sales are aggregated over all stores and departments, which makes errors lower than they would be at store or department level.
- Promotional markdown data is only recorded from November 2011, so the promotion feature may partly act as a proxy for the later period.
- Cost weights (under-forecast counted twice as heavily as over-forecast) are assumptions.

## Tools

Python, pandas, statsmodels, Prophet, XGBoost, scikit-learn, TensorFlow, matplotlib.
