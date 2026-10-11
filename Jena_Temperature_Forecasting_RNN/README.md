# Jena Temperature Forecasting: SimpleRNN vs LSTM vs GRU

A multivariate time-series project that forecasts the **air temperature 24 hours ahead** from the previous 72 hours of weather measurements. It compares SimpleRNN, LSTM and GRU models against a naive "tomorrow equals now" baseline.

📓 Notebook: `Jena_Temperature_Forecasting_RNN.ipynb`

---

## Dataset

- **Source:** Jena Climate 2009–2016, recorded at the Max Planck Institute for Biogeochemistry, downloaded automatically through `keras.utils.get_file`.
- **Size:** 420,551 readings taken every 10 minutes, with 14 weather variables (pressure, temperature, humidity, vapour pressure, wind speed and direction, air density, and others). There are no missing values.
- **Cleaning:** wind speed columns contain the placeholder value `-9999`. It is replaced with `0` before modelling.
- **Resampling:** only every 6th row is kept, giving **hourly data (70,091 rows)**.

## Workflow

1. **EDA:** shape, dtypes, duplicates, missing values, time-step regularity and the `-9999` sentinel check.
2. **Visualization:** full temperature series, a zoom on the first 7 days, monthly boxplots (seasonality), and a correlation heatmap.
3. **Preprocessing:**
   - Resample to hourly
   - Standardize each feature using the mean and std of the **first 70% (training period) only**
   - Build sliding windows: **72 hours of history (14 features) → temperature 24 hours ahead**, giving 69,995 samples
   - **Chronological** split into 70 / 15 / 15 train / validation / test (48,996 / 10,499 / 10,500), with no shuffling, to avoid look-ahead leakage
4. **Naive baseline:** predict that the temperature 24 h from now equals the last observed temperature.
5. **Models** (one recurrent layer with 64 units followed by `Dense(1)`):
   - **SimpleRNN** (about 5.1K parameters)
   - **LSTM** with Dropout 0.2 (about 20.3K parameters)
   - **GRU** with Dropout 0.2 (about 15.4K parameters)
6. **Evaluation:** predictions are converted back to °C and scored with MAE, RMSE and R². There are also actual-vs-predicted plots.

All models use Adam, MSE loss and early stopping on validation loss (patience 3).

## Results (test set, in °C)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **GRU** | **2.26** | **2.90** | **0.861** |
| LSTM | 2.30 | 2.94 | 0.857 |
| SimpleRNN | 2.39 | 3.07 | 0.844 |
| Naive baseline | 2.50 | 3.25 | 0.826 |

**Takeaway:** the GRU is the best model and also lighter than the LSTM. The margins are small, though: the best model improves on the naive baseline by only about **0.23 °C in MAE (roughly 9%)**. A 24-hour-ahead forecast is hard to beat because daily temperature cycles make "same time yesterday" a strong baseline. This is why a naive baseline is included, so the models are judged against something meaningful.

## Getting started

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
jupyter notebook Jena_Temperature_Forecasting_RNN.ipynb
```

Run all cells in order. The dataset (about 13 MB zipped) downloads on the first run, so an internet connection is required. The random seed is fixed (`SEED = 42`).

## Limitations and ideas

- The notebook does not save the trained model.
- Possible next steps: add time features (hour of day, day of year) with sine/cosine encoding, try Conv1D or stacked/bidirectional recurrent layers, and compare against a "same time yesterday" baseline.
- Further options are multi-step forecasting (predict all 24 future hours) and probabilistic forecasts with uncertainty intervals.
