import pandas as pd
import numpy as np
from datetime import timedelta
import os


def calculate_time_interval(df: pd.DataFrame) -> timedelta:
    """
    Calculate the average time interval between data points.

    :param df: DataFrame with DatetimeIndex.
    :return: Average time interval (timedelta type).
    """
    time_diffs = df.index.to_series().diff().dropna()
    avg_interval = time_diffs.mean()
    return avg_interval


def calculate_max_drawdown(cumulative_return: pd.Series) -> float:
    """
    Calculate the maximum drawdown of the equity curve.

    :param cumulative_return: Equity curve (Series).
    :return: Maximum drawdown (negative float representing percentage loss).
    """
    # Ensure series doesn't contain NaN values
    cumulative_return = cumulative_return.dropna()
    if len(cumulative_return) < 2:
        return 0.0

    # Calculate cumulative maximum
    cumulative_max = cumulative_return.cummax()
    # Calculate drawdown
    drawdown = (cumulative_max - cumulative_return) / cumulative_max
    # Get maximum drawdown
    max_drawdown = drawdown.max()

    # Ensure max drawdown is a finite number
    if np.isnan(max_drawdown) or np.isinf(max_drawdown):
        return 0.0

    # Return as negative value to represent loss
    return -max_drawdown


def calculate_annualization_factor(interval: timedelta) -> float:
    """
    Calculate annualization factor based on time interval.

    :param interval: Average time interval between data points.
    :return: Annualization factor (number of periods per year).
    """
    if interval <= timedelta(minutes=1):
        periods_per_year = 365 * 24 * 60  # Minutes in a year
    elif interval <= timedelta(hours=1):
        periods_per_year = 365 * 24  # Hours in a year
    elif interval <= timedelta(hours=4):
        periods_per_year = 365 * 6  # 4-hour periods in a year (6 periods per day)
    elif interval <= timedelta(days=1):
        periods_per_year = 365  # Days in a year
    elif interval <= timedelta(weeks=1):
        periods_per_year = 52  # Weeks in a year
    else:
        periods_per_year = 12  # Default to months
    return periods_per_year


def calculate_sharpe_ratio(returns: pd.Series, interval: timedelta, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio.

    :param returns: Periodic returns series.
    :param interval: Average time interval between data points.
    :param risk_free_rate: Annualized risk-free rate (default 0.0).
    :return: Sharpe ratio.
    """
    # Ensure series doesn't contain NaN values and infinity
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 2:
        print("Warning: Return series too short, cannot calculate Sharpe ratio.")
        return 0.0

    # Calculate mean return and standard deviation
    mean_return = returns.mean()
    std_dev = returns.std()

    # Check if standard deviation is valid
    if np.isnan(std_dev) or std_dev <= 1e-10:
        print(f"Warning: Return standard deviation too small or NaN: {std_dev}, cannot calculate Sharpe ratio.")
        return 0.0

    # Calculate annualization factor and Sharpe ratio
    periods_per_year = calculate_annualization_factor(interval)
    annualized_return = mean_return * periods_per_year
    annualized_std_dev = std_dev * np.sqrt(periods_per_year)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_std_dev

    # Ensure result is finite
    if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
        print(f"Warning: Sharpe ratio calculation result invalid: {sharpe_ratio}")
        return 0.0

    return sharpe_ratio


def calculate_sortino_ratio(returns: pd.Series, interval: timedelta, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sortino ratio.

    :param returns: Periodic returns series.
    :param interval: Average time interval between data points.
    :param risk_free_rate: Annualized risk-free rate (default 0.0).
    :return: Sortino ratio.
    """
    # Ensure series doesn't contain NaN values and infinity
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 2:
        print("Warning: Return series too short, cannot calculate Sortino ratio.")
        return 0.0

    # Calculate mean return
    mean_return = returns.mean()

    # Calculate downside deviation
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        print("Warning: No negative returns, cannot calculate downside deviation. Using standard deviation instead.")
        downside_std = returns.std()
    else:
        downside_std = downside_returns.std()

    # Check if downside standard deviation is valid
    if np.isnan(downside_std) or downside_std <= 1e-10:
        print(f"Warning: Downside standard deviation too small or NaN: {downside_std}, cannot calculate Sortino ratio.")
        return 0.0

    # Calculate annualization factor and Sortino ratio
    periods_per_year = calculate_annualization_factor(interval)
    annualized_return = mean_return * periods_per_year
    annualized_downside_std = downside_std * np.sqrt(periods_per_year)
    sortino_ratio = (annualized_return - risk_free_rate) / annualized_downside_std

    # Ensure result is finite
    if np.isnan(sortino_ratio) or np.isinf(sortino_ratio):
        print(f"Warning: Sortino ratio calculation result invalid: {sortino_ratio}")
        return 0.0

    return sortino_ratio


def calculate_calmar_ratio(cumulative_return: pd.Series, interval: timedelta) -> float:
    """
    Calculate Calmar ratio.

    :param cumulative_return: Equity curve.
    :param interval: Average time interval between data points.
    :return: Calmar ratio.
    """
    # Ensure series doesn't contain NaN values
    cumulative_return = cumulative_return.dropna()
    if len(cumulative_return) < 2:
        print("Warning: Equity curve too short, cannot calculate Calmar ratio.")
        return 0.0

    # Calculate total return
    starting_value = cumulative_return.iloc[0]
    ending_value = cumulative_return.iloc[-1]

    # Avoid division by zero
    if starting_value == 0:
        print("Warning: Starting value is zero, cannot calculate total return rate.")
        return 0.0

    total_return = (ending_value - starting_value) / starting_value

    # Calculate time span (years)
    time_span_years = (cumulative_return.index[-1] - cumulative_return.index[0]).days / 365.25
    if time_span_years < 0.01:  # Need at least a few days of data
        print("Warning: Time span too short, cannot calculate Calmar ratio.")
        return 0.0

    # Calculate annualized return
    annualized_return = (1 + total_return) ** (1 / time_span_years) - 1

    # Calculate maximum drawdown
    max_drawdown = calculate_max_drawdown(cumulative_return)
    if max_drawdown >= -1e-10:  # Since max_drawdown is now negative, we check if it's greater than -1e-10
        print(f"Warning: Maximum drawdown too small: {max_drawdown}, cannot calculate Calmar ratio.")
        return 0.0

    # Calculate Calmar ratio - use absolute value of max_drawdown since it's now negative
    calmar_ratio = annualized_return / abs(max_drawdown)

    # Ensure result is finite
    if np.isnan(calmar_ratio) or np.isinf(calmar_ratio):
        print(f"Warning: Calmar ratio calculation result invalid: {calmar_ratio}")
        return 0.0

    return calmar_ratio


def calculate_metrics(df: pd.DataFrame, risk_free_rate: float = 0.0) -> dict:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex type.")
    if 'cumulative_return' not in df.columns:
        raise ValueError("DataFrame must contain 'cumulative_return' column.")

    # Print data statistics for diagnosis
    print(f"Data statistics:")
    print(f"Number of data points: {len(df)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Cumulative return range: {df['cumulative_return'].min()} to {df['cumulative_return'].max()}")

    # Calculate time interval
    interval = calculate_time_interval(df)
    print(f"Average time interval: {interval}")

    # Calculate return changes - properly handle cases with initial value of 0
    # Use cumulative return equity curve instead of original total_account_value
    # Equity = 1 + cumulative_return
    equity_curve = 1 + df['cumulative_return']
    returns = equity_curve.pct_change().dropna()

    # Check calculated returns
    print(f"Returns sample: {returns.head()}")

    # Filter infinity and NaN values
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Filtered returns statistics: mean={returns.mean():.6f}, std={returns.std():.6f}, samples={len(returns)}")

    # Calculate metrics
    max_drawdown = calculate_max_drawdown(equity_curve)
    sharpe_ratio = calculate_sharpe_ratio(returns, interval, risk_free_rate)
    sortino_ratio = calculate_sortino_ratio(returns, interval, risk_free_rate)
    calmar_ratio = calculate_calmar_ratio(equity_curve, interval)

    return {
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio
    }

#
# # Data processing
# print("Starting to read data...")
# # Use full path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# csv_file = os.path.join(current_dir, 'cumulative_returns_SampleStrategy_20250319_043423.csv')
# print(f"Attempting to read file: {csv_file}")
# data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
# print(f"Read {len(data)} rows of data.")
#
# # Print column names for diagnosis
# print(f"Data columns: {data.columns.tolist()}")
#
# # Process data columns - ensure we have appropriate cumulative_return column
# if 'cumulative_return' not in data.columns:
#     if 'total_account_value' in data.columns:
#         # Check initial account value
#         initial_value = data['total_account_value'].iloc[0]
#         if initial_value <= 0:
#             print(f"Warning: Initial account value abnormal: {initial_value}, using first positive value")
#             # Find first positive value
#             for i, value in enumerate(data['total_account_value']):
#                 if value > 0:
#                     initial_value = value
#                     break
#             if initial_value <= 0:
#                 raise ValueError("Cannot find valid initial account value")
#
#         # Create cumulative return column
#         data['cumulative_return'] = (data['total_account_value'] - initial_value) / initial_value
#         print(f"Created cumulative_return column based on total_account_value. Initial value: {initial_value}")
#     else:
#         raise ValueError("Data contains neither cumulative_return nor total_account_value column.")
#
# # Check if cumulative return column is reasonable
# if data['cumulative_return'].isna().any():
#     print(f"Warning: cumulative_return column contains {data['cumulative_return'].isna().sum()} NaN values")
#     data['cumulative_return'] = data['cumulative_return'].fillna(0)
#
# if np.isinf(data['cumulative_return']).any():
#     print(f"Warning: cumulative_return column contains infinity values")
#     data['cumulative_return'] = data['cumulative_return'].replace([np.inf, -np.inf], np.nan).fillna(0)
#
# # Calculate metrics
# print("\nCalculating financial metrics...")
# metrics = calculate_metrics(data, risk_free_rate=0)
#
# print("\nFinancial metrics calculation results:")
# for metric, value in metrics.items():
#     print(f"{metric}: {value:.4f}")
