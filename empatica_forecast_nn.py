##################################### Imports ################################################
# Standard data science and ML libraries
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

# Command-line reading and data importing
import argparse
from pathlib import Path

# Data processing
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Standard library, used to store best model state
import copy

##################################### Model architectures ####################################
class ForecastRNN(torch.nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, horizon=7):
        super().__init__()
        self.rnn = torch.nn.RNN(input_size=1, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True)
        
        self.fc = torch.nn.Linear(hidden_size, horizon)  # <-- output H values

    def forward(self, x):
        _, h_n = self.rnn(x)   # h_n: (num_layers, batch, hidden_size)
        last_hidden = h_n[-1]        # take top layer: (batch, hidden_size)
        return self.fc(last_hidden)  # (batch, horizon)

class ForecastLSTM(torch.nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, horizon=7):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=hidden_size,
                                   num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, horizon)

    def forward(self, x):
        _, (h_n, c_n) = self.lstm(x)   # h_n, c_n: (num_layers, batch, hidden_size)
        last_hidden = h_n[-1]          # take top layer: (batch, hidden_size)
        return self.fc(last_hidden)    # (batch, horizon)

class ForecastGRU(torch.nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, horizon=7):
        super().__init__()
        self.gru = torch.nn.GRU(input_size=1, hidden_size=hidden_size,
                                   num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, horizon)

    def forward(self, x):
        _, h_n, = self.gru(x)   # h_n, c_n: (num_layers, batch, hidden_size)
        last_hidden = h_n[-1]          # take top layer: (batch, hidden_size)
        return self.fc(last_hidden)    # (batch, horizon)

class ForecastMLP(torch.nn.Module):
    def __init__(self, window_size: int, hidden_size: int, num_hidden_layers: int, horizon: int = 7, dropout: float = 0.0):
        super().__init__()
        layers = [torch.nn.Linear(window_size, hidden_size), torch.nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU()]
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(hidden_size, horizon))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, window_size)
        return self.net(x)  # (batch, horizon)

################################# Alternative Loss Functions ################################
class VarianceWeightedMSE(torch.nn.Module):
    def __init__(self, variance_weight: float = 0.5):
        super().__init__()
        self.variance_weight = variance_weight

    def forward(self, pred, target):
        # pred, target: (batch, horizon)
        mse = torch.mean((pred - target) ** 2)

        # Compare variance of predictions vs variance of targets, per sample
        pred_var = pred.var(dim=1, unbiased=False)      # (batch,)
        target_var = target.var(dim=1, unbiased=False)  # (batch,)

        # Penalize when pred_var < target_var (flatlining relative to actual variation)
        # Only penalize the shortfall, not cases where pred is already more variable
        variance_penalty = torch.clamp(target_var - pred_var, min=0).mean()

        return mse + self.variance_weight * variance_penalty

class DiffWeightedMSE(torch.nn.Module):
    def __init__(self, diff_weight: float = 0.5):
        super().__init__()
        self.diff_weight = diff_weight

    def forward(self, pred, target):
        mse = torch.mean((pred - target) ** 2)

        pred_diff = pred[:, 1:] - pred[:, :-1]        # (batch, horizon-1)
        target_diff = target[:, 1:] - target[:, :-1]  # (batch, horizon-1)

        diff_mse = torch.mean((pred_diff - target_diff) ** 2)

        return mse + self.diff_weight * diff_mse

##################################### Data Loading ##########################################
# TODO: set dictionary key names to your biomarker folder names
def load_data(subject: str, biomarker: str, root: Path):
    full_data_path = root / f"{subject}_{biomarker}"

    # TODO: update key names to match your biomarker folders
    csv_biomarker_names = {
        "activity_counts": "activity_counts",
        "heart_rates": "pulse_rate_bpm",
        "prvs": "prv_rmssd_ms"
    }

    csv_biomarker = csv_biomarker_names[biomarker]

    data = pd.concat(
    [pd.read_csv(f) for f in full_data_path.glob("*.csv")],
    ignore_index=True
    )

    # Convert timestamp_iso from string to pandas datetime type
    data = (data.assign(timestamp_iso=pd.to_datetime(data["timestamp_iso"])))

    daily_df = (
    data
    .assign(day=data["timestamp_iso"].dt.floor("D").dt.date)
    .groupby("day", as_index=False)[csv_biomarker]
    .mean()
    .rename(columns={csv_biomarker: "biomarker_avg"})
    )

    daily_df['day'] = pd.to_datetime(daily_df['day'])

    ####################### Missing Data Handling ###################
    daily_df = daily_df.set_index('day').sort_index()

    # Force a complete daily date range
    full_range = pd.date_range(daily_df.index.min(), daily_df.index.max(), freq='D')
    daily_df = daily_df.reindex(full_range)
    daily_df.index.name = 'day'

    # Linearly interpolate all gaps, index-aware
    daily_df["biomarker_avg"] = daily_df["biomarker_avg"].interpolate(
        method='time', limit_direction='both'
    )

    # --- Verify no gaps remain, checked against the real DatetimeIndex ---
    diffs = daily_df.index.to_series().diff().dropna()
    gap_locs = diffs[diffs != pd.Timedelta(days=1)]
    assert gap_locs.empty, f"Gaps remain in date index:\n{gap_locs}"

    assert daily_df['biomarker_avg'].isna().sum() == 0, "NaNs remain in biomarker_avg"

    daily_df = daily_df.reset_index()

    return daily_df

##################################### Data Processing #######################################
def split_loaders(act_daily_df, train_start_str: str, test_start_str: str, test_end_str: str, 
                  window_size: int, horizon: int = 7, add_channel_dim: bool = True):
    
    values = act_daily_df["biomarker_avg"].to_numpy()
    dates = act_daily_df["day"].to_numpy()

    """
                        TRAIN PHASE START DATES
                      SPT04:              SPT05:
    Rehab:            2026-01-23          2026-02-18
    Implant:          2026-03-16          2026-04-16
    Implant Recovery: 2026-03-23          2026-04-13
    Optimization:     2026-03-30          2026-04-20
    Implant + Rehab:  2026-04-20          2026-05-11
    Post-Rehab:       2026-05-18          2026-06-22
    """

    # Convert string date inputs to date objects
    train_start_date = pd.to_datetime(train_start_str)
    test_start_date = pd.to_datetime(test_start_str)
    test_end_date = pd.to_datetime(test_end_str)

    def find_date_idx(dates, target_date, label):
        matches = np.where(dates == np.datetime64(target_date))[0]
        if len(matches) == 0:
            raise ValueError(f"{label} date {target_date} not found in data range "
                            f"[{dates.min()} to {dates.max()}]")
        return matches[0]

    # Get indices of start and end dates
    train_start = find_date_idx(dates, train_start_date, "train_start")
    test_start = find_date_idx(dates, test_start_date, "test_start")
    test_end = find_date_idx(dates, test_end_date, "test_end") + 1

    # Fully define train, val, and test sets, 80-20 train-val split
    n = len(values[train_start : test_start])
    val_start = train_end = int(0.8 * n) + train_start
    val_end = test_start

    # Rescale data to handle vanishing/exploding gradients
    scaler = StandardScaler()
    scaler.fit(values[:train_end].reshape(-1, 1))
    values_scaled = scaler.transform(values.reshape(-1, 1)).flatten()

    def make_windows(start, end):
        # Each element of X is a {window_size}-day ordered window of daily act_count averages
        # Each element of y is an ordered horizon of the NEXT 7 act_count avgs after the window
        X, y = [], []
        for i in range(start, end):
            X.append(values_scaled[i : i + window_size])
            y.append(values_scaled[i + window_size : i + window_size + horizon])
        X = torch.tensor(np.array(X), dtype=torch.float32)
        if add_channel_dim:
            X = X.unsqueeze(-1)  # (n, window_size, 1) for RNN
        y = torch.tensor(np.array(y), dtype=torch.float32)
        return X, y

    ############################## Training Set #############################
    # Train set: all windows from beginning until val_start
    X_train, y_train = make_windows(train_start, train_end - window_size - horizon + 1)

    ############################## Validation Set #############################
    X_val, y_val = make_windows(val_start - window_size, val_end - window_size - horizon + 1)
    if len(X_val) == 0:
        raise ValueError("Train set is not long enough for validation; make train_start earlier or test_start later")

    ############################## Test Set #############################
    # Single window of input ending at test_start, horizon from test_start to test_end
    X_test = torch.tensor(values_scaled[test_start - window_size : test_start], dtype=torch.float32).unsqueeze(0)
    if add_channel_dim: X_test = X_test.unsqueeze(-1)  # (1, window_size, 1) for RNN; (1, window_size) for MLP

    y_test = torch.tensor(values_scaled[test_start : test_end], dtype=torch.float32).unsqueeze(0)

    ############################## Exporting Loaders and Dates #############################
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader  = DataLoader(TensorDataset(X_val,  y_val),  batch_size=1, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=1, shuffle=False)

    train_dates = dates[train_start : train_end - window_size - horizon + 1]
    val_dates = dates[val_start - window_size : val_end - window_size - horizon + 1]
    test_dates = dates[test_start : test_end]

    return train_loader, val_loader, test_loader, scaler, train_dates, val_dates, test_dates

############################ Model evaluation helper functions ###############################
def calculate_mse(predictions, actual):
    sq_error = (actual - predictions)**2
    return sq_error.mean()

def calculate_rmse(predictions, actual):
    sq_error = (actual - predictions)**2
    return np.sqrt(sq_error.mean())

def calculate_mape(predictions, actual):
    abs_pct_error = 100 * np.abs(actual - predictions) / actual
    return abs_pct_error.mean()

############################## Model Training and Validation ################################
def train(model, train_loader: DataLoader, val_loader: DataLoader, 
          loss_fn, optimizer, scaler, num_epochs: int, error_metric: str = 'RMSE'):
    best_val_error = float("inf")
    best_model_state = None

    train_errors, val_errors = [], []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_preds, train_true = [], []
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            y_pred = model.forward(x_batch)  # Model forward pass predictions
            loss = loss_fn(y_pred, y_batch)  # Compute MSE or MSE variant loss
            loss.backward()                  # Backprop through the model
            optimizer.step()                 # Update model weights

            # Detach before collecting, so we don't carry the graph around
            train_preds.append(y_pred.detach().numpy())
            train_true.append(y_batch.detach().numpy())

        # Rescale predictions and actuals to original scale before calculating and recording RMSE
        y_train_pred_inv = scaler.inverse_transform(np.concatenate(train_preds).reshape(-1, 1)).flatten()
        y_train_inv = scaler.inverse_transform(np.concatenate(train_true).reshape(-1, 1)).flatten()

        if error_metric == 'MAPE':
            train_error = calculate_mape(y_train_pred_inv, y_train_inv)
        else: # error_metric == 'RMSE'
            train_error = calculate_rmse(y_train_pred_inv, y_train_inv)

        train_errors.append(train_error)

        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                y_val_pred = model(x_val)
                val_preds.append(y_val_pred.numpy())
                val_true.append(y_val.numpy())
        
        # Rescale predictions and actuals to original scale before calculating and recording RMSE
        y_val_pred_inv = scaler.inverse_transform(np.concatenate(val_preds).reshape(-1, 1)).flatten()
        y_val_inv = scaler.inverse_transform(np.concatenate(val_true).reshape(-1, 1)).flatten()

        if error_metric == 'MAPE':
            val_error = calculate_mape(y_val_pred_inv, y_val_inv)
        else: # error_metric == 'RMSE'
            val_error = calculate_rmse(y_val_pred_inv, y_val_inv)

        val_errors.append(val_error)

        if val_error < best_val_error:
            best_val_error = val_error
            best_model_state = copy.deepcopy(model.state_dict())  

    # Early stopping: Load the best-performing model (on validation data) at the end of training
    model.load_state_dict(best_model_state)

    return train_errors, val_errors

#################################### Loss-Epoch Plotting ####################################
def loss_plot(train_losses: list[float], val_losses: list[float], config: dict):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(len(train_losses)), train_losses, label='Train')
    ax.plot(range(len(val_losses)), val_losses, label='Val', linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE Loss")
    ax.set_title(f"""RMSE Loss by Epoch \n 
                    Prompt: '% python empatica_forecast_nn.py {config["model_type"]} {config["subject"]} 
                    {config["train_start"]} {config["test_start"]} {config["test_end"]} {config["window_size"]} 
                    {config["hidden_size"]} {config["num_layers"]} {config["lr"]} {config["num_epochs"]}' """)
    ax.legend()
    plt.tight_layout()
    plt.show()

#################################### Forecast Plotting ######################################
def evaluate_and_plot(model, loader: DataLoader, scaler, dates, 
                      split_name: str, config: dict, error_metric: str):
    model.eval()
    all_preds, all_actuals = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            y_pred = model(x_batch)
            all_preds.append(y_pred)
            all_actuals.append(y_batch)

    y_pred   = torch.cat(all_preds,   dim=0).squeeze().numpy()
    y_actual = torch.cat(all_actuals, dim=0).squeeze().numpy()

    # Restore predictions and actual values to original scale (un-normalize)
    y_pred   = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
    y_actual = scaler.inverse_transform(y_actual.reshape(-1, 1)).reshape(y_actual.shape)

    if error_metric == 'MAPE':
        error = calculate_mape(y_pred, y_actual)
    else: # error_metric == 'RMSE':
        error = calculate_rmse(y_pred, y_actual)

    # If using a multi-step horizon: plot only the first forecast step to avoid overlap issues
    if y_pred.ndim > 1:
        y_pred_plot   = y_pred[:, 0]
        y_actual_plot = y_actual[:, 0]
    else:
        y_pred_plot   = y_pred
        y_actual_plot = y_actual

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, y_actual_plot, label="Actual")
    ax.plot(dates, y_pred_plot,   label="Predicted", linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{config["biomarker"]} average")
    ax.set_title(f"""{config["subject"]} Actual vs Predicted {config["biomarker"]} ({split_name}). 
                    {error_metric}: {error:.2f} \n Prompt: {config["prompt"]} """)
    ax.legend()
    plt.tight_layout()
    plt.show()

    return y_pred, y_actual, error

# TODO: Set data root, horizon, and eval metric
def main(model_type: str, subject: str, biomarker: str, 
         train_start: str, test_start: str, test_end: str, 
         window_size: int, hidden_size: int, num_layers: int, lr: float, num_epochs: int):
    """
    model_type: Which model to train: {RNN, LSTM, GRU, MLP}
    subject: Which subject to download data and model: {SPT04, SPT05}
    biomarker: Which biomarker to forecast. Match to name of your csvs folder.
    train_start: First day of data to include in training set. 'YYYY-MM-DD' format.
    test_start: First day of the test window. 'YYYY-MM-DD' format.
    test_end: Last day of the test window, inclusive. Must align with model horizon (7 days by default).
    window_size: How many days of lookback the model uses to predict the next seven days.
    hidden_size: Hidden layer size 
    num_layers: Number of hidden layers
    lr: Learning rate of the optimizer
    num_epochs: How many epochs to train
    """

    # TODO: Set DATA_ROOT to the path to the root directory of your data 
    DATA_ROOT = Path("Empatica_data/inputs/")

    # TODO: Set HORIZON to the number of days you wish to forecast per training window
    HORIZON = 7

    # TODO: Set ERROR_METRIC to the error metric you wish to display on plots: {RMSE, MAPE}
    ERROR_METRIC = 'MAPE'

    assert ((ERROR_METRIC == 'MAPE') or (ERROR_METRIC == 'RMSE')), "Set ERROR_METRIC to either RMSE or MAPE"

    config = {
        "model_type": model_type,
        "subject": subject,
        "biomarker": biomarker,
        "train_start": train_start,
        "test_start": test_start,
        "test_end": test_end,
        "window_size": window_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "lr": lr,
        "num_epochs": num_epochs,
        "prompt": f'''% python empatica_forecast_nn.py {model_type} {subject} {biomarker}
                    {train_start} {test_start} {test_end} 
                    {window_size} {hidden_size} {num_layers} {lr} {num_epochs}
                '''
    }
    
    daily_df = load_data(subject, biomarker, DATA_ROOT)

    # All recurrent models require a channel dimension in their input, while MLP does not
    channel_dim = True

    # Initialize model with hidden_size and num_layers provided by command line
    if model_type == 'RNN':
        model = ForecastRNN(hidden_size=hidden_size, num_layers=num_layers, horizon=HORIZON)
    elif model_type == 'LSTM':
        model = ForecastLSTM(hidden_size=hidden_size, num_layers=num_layers, horizon=HORIZON)
    elif model_type == 'GRU':
        model = ForecastGRU(hidden_size=hidden_size, num_layers=num_layers, horizon=HORIZON)
    elif model_type == 'MLP':
        model = ForecastMLP(window_size=window_size, hidden_size=hidden_size, 
                            num_hidden_layers=num_layers, horizon=HORIZON)
        channel_dim = False

    (train_loader, val_loader, test_loader, scaler, train_dates, val_dates, test_dates) = split_loaders(
        daily_df, 
        train_start, 
        test_start, 
        test_end, 
        window_size, 
        horizon=HORIZON, 
        add_channel_dim=channel_dim
    )

    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    # Alternative loss functions, performance not found to be better but possibly worth working with:
    # loss_fn = VarianceWeightedMSE(variance_weight=)
    # loss_fn = DiffWeightedMSE(diff_weight=)

    # Train and validate model, storing errors for loss plotting
    train_errors, val_errors = train(model, train_loader, val_loader, loss_fn, optimizer, scaler, num_epochs)

    # Plot training and validation loss against epoch to determine if model is properly training and/or overfitting
    loss_plot(train_errors, val_errors, config)

    # Plot final model predictions against actual values for each split
    evaluate_and_plot(model, train_loader, scaler, train_dates, "Training", config, ERROR_METRIC)
    evaluate_and_plot(model, val_loader, scaler, val_dates, "Validation", config, ERROR_METRIC)
    evaluate_and_plot(model, test_loader, scaler, test_dates, "Testing", config, ERROR_METRIC)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", type=str, help="RNN, LSTM, GRU, or MLP")
    parser.add_argument("subject", type=str, help="Subject code: SPT04 or SPT05")
    parser.add_argument("biomarker", type=str, help="activity_counts, prvs, heart_rates, etc")
    parser.add_argument("train_start", type=str, help="Start date of training data, in 'YYYY-MM-DD' format")
    parser.add_argument("test_start", type=str, help="Start date of test window, in 'YYYY-MM-DD' format")
    parser.add_argument("test_end", type=str, help="End date of test window (inclusive), in 'YYYY-MM-DD' format")
    parser.add_argument("window_size", type=int, help="How many input days to include for each prediction")
    parser.add_argument("hidden_size", type=int, help="Hidden layer size of LSTM")
    parser.add_argument("num_layers", type=int, help="Number of LSTM hidden layers")
    parser.add_argument("lr", type=float, help="Learning rate")
    parser.add_argument("num_epochs", type=int, help="How many training epochs to run")
    
    args = parser.parse_args()

    main(args.model_type, args.subject, args.biomarker, 
         args.train_start, args.test_start, args.test_end, 
         args.window_size, args.hidden_size, args.num_layers, args.lr, args.num_epochs)
