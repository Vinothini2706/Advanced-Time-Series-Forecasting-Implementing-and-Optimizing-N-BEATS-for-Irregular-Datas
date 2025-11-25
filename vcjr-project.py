import math
import random
import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor


# -------------------------------------------------------
# 1. Synthetic irregular time series generator
# -------------------------------------------------------
def generate_synthetic_series(
    n_points: int = 6000,
    freq: str = "H",
    seed: int = 42,
    missing_ratio: float = 0.1,
) -> pd.DataFrame:
    """
    Generate a synthetic time series with:
    - multiple seasonalities (daily & weekly)
    - trend + level shift
    - noise
    - irregular sampling via random missing timestamps
    """
    np.random.seed(seed)
    random.seed(seed)

    start = pd.Timestamp("2020-01-01")
    full_index = pd.date_range(start=start, periods=n_points, freq=freq)
    t = np.arange(n_points)

    # Daily (period 24) and weekly (24*7) seasonalities
    daily = 5 * np.sin(2 * np.pi * t / 24)
    weekly = 3 * np.sin(2 * np.pi * t / (24 * 7))

    # Slow linear trend
    trend = 0.002 * t

    # Level shift at 2/3 of series
    shift_point = int(n_points * 2 / 3)
    level_shift = np.zeros(n_points)
    level_shift[shift_point:] = 10

    # Gaussian noise
    noise = np.random.normal(0, 1.5, size=n_points)

    values = 20 + daily + weekly + trend + level_shift + noise
    df = pd.DataFrame({"timestamp": full_index, "value": values})
    df.set_index("timestamp", inplace=True)

    # Make series irregular: remove random timestamps
    n_missing = int(missing_ratio * n_points)
    missing_idx = np.random.choice(np.arange(n_points), size=n_missing, replace=False)
    df_irregular = df.drop(df.index[missing_idx]).sort_index()

    # Reindex back to regular grid and interpolate
    df_full = df_irregular.reindex(full_index)
    df_full["value"] = df_full["value"].interpolate().ffill().bfill()

    df_full.reset_index(inplace=True)
    df_full.rename(columns={"index": "timestamp"}, inplace=True)
    return df_full


# -------------------------------------------------------
# 2. N-BEATS model (PyTorch)
# -------------------------------------------------------
class NBeatsBlock(nn.Module):
    """
    Simple generic N-BEATS block.
    """

    def __init__(self, input_length, forecast_length, hidden_dim=256, n_layers=4):
        super().__init__()
        layers = []
        in_features = input_length
        for _ in range(n_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim
        self.fc = nn.Sequential(*layers)
        self.backcast_head = nn.Linear(hidden_dim, input_length)
        self.forecast_head = nn.Linear(hidden_dim, forecast_length)

    def forward(self, x):
        h = self.fc(x)
        backcast = self.backcast_head(h)
        forecast = self.forecast_head(h)
        return backcast, forecast


class NBeats(nn.Module):
    """
    Stacked generic N-BEATS.
    """

    def __init__(
        self,
        input_length: int,
        forecast_length: int,
        n_blocks: int = 4,
        hidden_dim: int = 256,
        n_layers: int = 4,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                NBeatsBlock(input_length, forecast_length, hidden_dim, n_layers)
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x):
        residual = x
        forecast_sum = 0
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast
            forecast_sum = forecast_sum + forecast
        return forecast_sum


class WindowDataset(Dataset):
    """
    Turn a 1D array into (input_window -> forecast_window) pairs.
    """

    def __init__(self, series: np.ndarray, input_length: int, forecast_length: int):
        self.series = series.astype(np.float32)
        self.input_length = input_length
        self.forecast_length = forecast_length
        self.max_idx = len(series) - input_length - forecast_length

    def __len__(self):
        return self.max_idx

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.input_length]
        y = self.series[idx + self.input_length : idx + self.input_length + self.forecast_length]
        return x, y


# -------------------------------------------------------
# 3. Metrics
# -------------------------------------------------------
@dataclass
class Metrics:
    mae: float
    rmse: float
    mape: float
    mase: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, seasonal_period: int = 1) -> Metrics:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    eps = 1e-6
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100

    if seasonal_period <= 0 or seasonal_period >= len(y_true):
        seasonal_period = 1
    naive_forecast = y_true[:-seasonal_period]
    naive_actual = y_true[seasonal_period:]
    naive_mae = mean_absolute_error(naive_actual, naive_forecast)
    mase = mae / (naive_mae + eps)

    return Metrics(mae=mae, rmse=rmse, mape=mape, mase=mase)


# -------------------------------------------------------
# 4. N-BEATS training + Optuna hyperparameter tuning
# -------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def eval_epoch(model, loader, criterion):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def run_nbeats_training(
    train_series,
    val_series,
    input_length,
    forecast_length,
    n_blocks,
    hidden_dim,
    n_layers,
    lr,
    weight_decay,
    batch_size=64,
    epochs=20,
):
    train_ds = WindowDataset(train_series, input_length, forecast_length)
    val_ds = WindowDataset(val_series, input_length, forecast_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = NBeats(
        input_length=input_length,
        forecast_length=forecast_length,
        n_blocks=n_blocks,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = eval_epoch(model, val_loader, criterion)
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
        print(f"Epoch {epoch+1}/{epochs} - train {train_loss:.4f}, val {val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val


def tune_nbeats_with_optuna(train_series, val_series, input_length, forecast_length, n_trials=15):
    def objective(trial):
        n_blocks = trial.suggest_int("n_blocks", 2, 6)           # number of stacks
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
        n_layers = trial.suggest_int("n_layers", 2, 4)
        lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)          # learning rate
        weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        _, val_loss = run_nbeats_training(
            train_series=train_series,
            val_series=val_series,
            input_length=input_length,
            forecast_length=forecast_length,
            n_blocks=n_blocks,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=15,
        )
        # Optuna minimizes RMSE on validation set
        return math.sqrt(val_loss)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("Best N-BEATS parameters:", study.best_params)
    print("Best validation RMSE:", study.best_value)
    return study.best_params


def forecast_with_nbeats(model, series, input_length, forecast_length):
    model.eval()
    series = series.astype(np.float32)
    preds = []
    with torch.no_grad():
        for i in range(len(series) - input_length - forecast_length + 1):
            x = series[i : i + input_length]
            x = torch.from_numpy(x).unsqueeze(0).to(DEVICE)
            out = model(x).cpu().numpy().ravel()
            preds.append(out[-1])  # last step of forecast horizon
    return np.array(preds)


# -------------------------------------------------------
# 5. Benchmarks: SARIMA and XGBoost
# -------------------------------------------------------
def run_sarima(train_values, test_len, seasonal_period=24):
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, seasonal_period)
    model = SARIMAX(
        train_values,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    forecast = res.forecast(steps=test_len)
    return np.asarray(forecast)


def make_lag_features(series: pd.Series, max_lag: int = 48):
    df = pd.DataFrame({"y": series})
    for lag in range(1, max_lag + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)

    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df.dropna(inplace=True)

    X = df.drop(columns=["y"])
    y = df["y"]
    return X, y


def run_xgboost(train_series: pd.Series, test_series: pd.Series, max_lag: int = 48):
    full_series = pd.concat([train_series, test_series])
    X, y = make_lag_features(full_series, max_lag=max_lag)

    split_timestamp = train_series.index[-1]
    train_mask = X.index <= split_timestamp

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return np.asarray(y_test), preds


# -------------------------------------------------------
# 6. Full pipeline
# -------------------------------------------------------
def main():
    # 1) Data generation
    df = generate_synthetic_series()
    df.set_index("timestamp", inplace=True)
    values = df["value"].values.reshape(-1, 1)

    n = len(values)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_values = values[:train_end]
    val_values = values[train_end:val_end]
    test_values = values[val_end:]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_values).ravel()
    val_scaled = scaler.transform(val_values).ravel()
    test_scaled = scaler.transform(test_values).ravel()

    # Independent look-back and forecast horizons
    input_length = 48       # look-back length
    forecast_length = 24    # forecast horizon

    # 2) Hyperparameter optimization for N-BEATS
    best_params = tune_nbeats_with_optuna(
        train_series=train_scaled,
        val_series=val_scaled,
        input_length=input_length,
        forecast_length=forecast_length,
        n_trials=10,
    )

    # 3) Final N-BEATS training on train+val
    train_plus_val = np.concatenate([train_scaled, val_scaled])
    model, _ = run_nbeats_training(
        train_series=train_plus_val,
        val_series=test_scaled[: len(val_scaled)],  # small dummy val
        input_length=input_length,
        forecast_length=forecast_length,
        n_blocks=best_params["n_blocks"],
        hidden_dim=best_params["hidden_dim"],
        n_layers=best_params["n_layers"],
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        batch_size=best_params["batch_size"],
        epochs=25,
    )

    # N-BEATS forecasts
    all_scaled = np.concatenate([train_plus_val, test_scaled])
    nbeats_pred_scaled = forecast_with_nbeats(
        model=model,
        series=all_scaled,
        input_length=input_length,
        forecast_length=forecast_length,
    )
    nbeats_pred_scaled = nbeats_pred_scaled[-len(test_scaled):]
    nbeats_pred = scaler.inverse_transform(nbeats_pred_scaled.reshape(-1, 1)).ravel()
    y_test = test_values.ravel()

    nbeats_metrics = compute_metrics(y_test, nbeats_pred, seasonal_period=24)
    print("\nN-BEATS Metrics:", nbeats_metrics)

    # 4) SARIMA benchmark
    sarima_pred = run_sarima(train_values.ravel(), test_len=len(test_values), seasonal_period=24)
    sarima_metrics = compute_metrics(y_test, sarima_pred, seasonal_period=24)
    print("\nSARIMA Metrics:", sarima_metrics)

    # 5) XGBoost benchmark
    train_index = df.index[:train_end]
    test_index = df.index[val_end:]
    y_train_series = pd.Series(train_values.ravel(), index=train_index)
    y_test_series = pd.Series(test_values.ravel(), index=test_index)

    xgb_y_true, xgb_pred = run_xgboost(y_train_series, y_test_series, max_lag=48)
    xgb_metrics = compute_metrics(xgb_y_true, xgb_pred, seasonal_period=24)
    print("\nXGBoost Metrics:", xgb_metrics)

    # 6) Comparison plot
    plt.figure(figsize=(12, 5))
    plt.plot(df.index[val_end:], y_test, label="Actual")
    plt.plot(df.index[val_end:val_end + len(nbeats_pred)], nbeats_pred, label="N-BEATS")
    plt.plot(df.index[val_end:val_end + len(sarima_pred)], sarima_pred, label="SARIMA")
    plt.legend()
    plt.title("Test Set Forecasts")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
