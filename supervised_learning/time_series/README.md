# Time Series Forecasting -  Bitcoin (BTC) Price Prediction

This project builds an RNN that uses the past 24 hours of BTC market
data to forecast the closing price of the following hour
(approximately how long the average BTC transaction historically took to
confirm).

## Files

| File | Description |
|------|-------------|
| `preprocess_data.py` | Cleans, resamples, scales, and windows the raw Coinbase/Bitstamp datasets. |
| `forecast_btc.py` | Builds, trains, and validates the Keras RNN using `tf.data.Dataset`. |
| `README.md` | This file. |

### The data

Both coinbaseUSD.csv and bitstampUSD.csv contain raw, per-minute
trading data with columns:

Timestamp, Open, High, Low, Close, Volume_(BTC), Volume_(Currency), Weighted_Price

#### Preprocessing decisions

Not every data point is useful. The raw files are per-minute and very
noisy, with gaps wherever no trade occurred in a given minute. Since the
forecasting task is explicitly hourly (24h in → 1h-ahead close out), the
per-minute rows are resampled into hourly OHLCV candles:

  Open = first per-minute open of the hour
  High = max per-minute high of the hour
  Low  = min per-minute low of the hour
  Close = last per-minute close of the hour
  Volume_(BTC) = sum of per-minute BTC volume over the hour
  Weighted_Price = the BTC-volume-weighted average of the per-minute
  VWAP for the hour (falls back to a plain mean on zero-volume hours)

This both removes a lot of minute-level noise and shrinks the input
sequence length the RNN has to learn dependencies over (24 steps per day,
instead of 1,440).

###### Not every feature is useful. 
 - Volume_(Currency) is dropped — it's redundant, being approximately
   Volume_(BTC) * Weighted_Price for that row, so it adds no new
   information for the model to learn from.
 - The raw Timestamp is dropped as a model feature (it's
   monotonically increasing and huge in magnitude, with no cyclical
   structure a plain RNN can exploit). It's used internally only to sort
   and align rows before being discarded.

###### Missing hours.
 Hours with no trades produce NaN OHLC values. Prices
are forward-filled (no trade ⇒ price didn't move) and volume is
zero-filled (no trade ⇒ zero BTC transacted). Any leading NaNs (before
the first trade in the dataset) are dropped outright, since there's no
sensible value to fill them with.

###### Combining exchanges.
Coinbase and Bitstamp hourly candles are
concatenated and sorted by time to give the model more historical signal.
Where both exchanges report the same hour, Coinbase's row is kept (it's
the more liquid / representative venue in the overlapping period).

Is the "current" window relevant? Yes — the model consumes the past
24 complete hourly candles as input and is trained to predict the
Close of the next hour (window[t-24:t] → close[t+1]). The
in-progress/current partial hour itself is never used as a feature, since
it is, by definition, an incomplete observation at prediction time.

Rescaling. Prices (thousands–tens of thousands of USD) and volumes
(fractions of a BTC) live on very different scales, which would bias an
RNN's gradient updates toward the larger-magnitude features. All features
are scaled to [0, 1] with min-max scaling. Critically, the scaler is
fit only on the training split of a chronological train/val/test
split (70% / 15% / 15%, in time order, never shuffled before splitting),
so no information about future (validation/test) price ranges leaks into
training.

###### How is the result saved? 
Everything is saved into a single compressed
btc_preprocessed.npz containing:
 - X_train, y_train, X_val, y_val, X_test, y_test — windowed,
scaled (N, 24, 6) inputs and (N,) targets, ready to be wrapped in a
tf.data.Dataset
 - data_min, data_range — the min-max scaler parameters fit on the
training data, saved so predictions can later be un-scaled back into
real USD if desired
 - feature_columns — the ordered feature names, for reference

This avoids re-doing the (relatively expensive) cleaning/resampling step
every time the model is trained or re-trained.

##### The model
forecast_btc.py builds a stacked LSTM network:
## Model Architecture

```text
Input (24, 6)
    ├── LSTM(64, return_sequences=True)
    ├── Dropout(0.2)
    ├── LSTM(32)
    ├── Dropout(0.2)
    ├── Dense(16, activation="relu")
    └── Dense(1)
```

- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Data Pipeline:** `tf.data.Dataset.from_tensor_slices`, shuffled (by window, not within a window) and batched for training; unshuffled for validation and testing.
- **Regularization:** Dropout between LSTM layers and early stopping based on validation loss to reduce overfitting.
