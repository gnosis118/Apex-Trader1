import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta

def format_number(number, precision=2):
    """
    Format a number with thousands separator and a specified precision
    
    Parameters:
    -----------
    number: float
        The number to format
    precision: int
        Number of decimal places to display
        
    Returns:
    --------
    str
        The formatted number as a string
    """
    return f"{number:,.{precision}f}"

def format_percentage(number, precision=2):
    """
    Format a number as a percentage with a specified precision
    
    Parameters:
    -----------
    number: float
        The number to format (e.g., 0.123 for 12.3%)
    precision: int
        Number of decimal places to display
        
    Returns:
    --------
    str
        The formatted percentage as a string
    """
    return f"{number * 100:.{precision}f}%"

def calculate_returns_metrics(returns_series):
    """
    Calculate various metrics from a returns series
    
    Parameters:
    -----------
    returns_series: pandas.Series
        Series containing return values
        
    Returns:
    --------
    dict
        Dictionary containing various return metrics
    """
    # Ensure we have valid data
    if returns_series is None or len(returns_series) == 0:
        return {
            'total_return': 0,
            'annualized_return': 0,
            'volatility': 0,
            'sharpe': 0,
            'sortino': 0,
            'max_drawdown': 0
        }
    
    # Remove NaN values
    returns = returns_series.dropna()
    
    if len(returns) == 0:
        return {
            'total_return': 0,
            'annualized_return': 0,
            'volatility': 0,
            'sharpe': 0,
            'sortino': 0,
            'max_drawdown': 0
        }
    
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate total return
    total_return = cum_returns.iloc[-1] - 1 if len(cum_returns) > 0 else 0
    
    # Calculate annualized return
    n_periods = len(returns)
    annualized_return = (1 + total_return) ** (252 / n_periods) - 1 if n_periods > 0 else 0
    
    # Calculate volatility
    volatility = returns.std() * np.sqrt(252)
    
    # Calculate Sharpe ratio
    risk_free_rate = 0.02  # Assume 2% risk-free rate
    excess_return = annualized_return - risk_free_rate
    sharpe = excess_return / volatility if volatility > 0 else 0
    
    # Calculate Sortino ratio
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = excess_return / downside_deviation if downside_deviation > 0 else 0
    
    # Calculate maximum drawdown
    rolling_max = cum_returns.expanding().max()
    drawdown = (cum_returns / rolling_max - 1)
    max_drawdown = drawdown.min()
    
    return {
        'total_return': total_return * 100,  # Convert to percentage
        'annualized_return': annualized_return * 100,  # Convert to percentage
        'volatility': volatility * 100,  # Convert to percentage
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': abs(max_drawdown) * 100  # Convert to positive percentage
    }

def calculate_win_rate(trades):
    """
    Calculate win rate and related metrics from a list of trades
    
    Parameters:
    -----------
    trades: list
        List of trade dictionaries with 'profit_pct' key
        
    Returns:
    --------
    dict
        Dictionary containing win rate and related metrics
    """
    if not trades:
        return {
            'win_rate': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'largest_win': 0,
            'largest_loss': 0
        }
    
    # Calculate wins and losses
    wins = [trade for trade in trades if trade['profit_pct'] > 0]
    losses = [trade for trade in trades if trade['profit_pct'] <= 0]
    
    # Calculate win rate
    win_rate = len(wins) / len(trades) * 100 if len(trades) > 0 else 0
    
    # Calculate profit factor
    total_wins = sum(trade['profit_pct'] for trade in wins) if wins else 0
    total_losses = sum(abs(trade['profit_pct']) for trade in losses) if losses else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    # Calculate average win and loss
    avg_win = total_wins / len(wins) if len(wins) > 0 else 0
    avg_loss = total_losses / len(losses) if len(losses) > 0 else 0
    
    # Calculate largest win and loss
    largest_win = max([trade['profit_pct'] for trade in wins]) if wins else 0
    largest_loss = min([trade['profit_pct'] for trade in losses]) if losses else 0
    
    return {
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'largest_win': largest_win,
        'largest_loss': largest_loss
    }

def save_strategy_results(strategy_name, results, directory='results'):
    """
    Save strategy backtest results to a JSON file
    
    Parameters:
    -----------
    strategy_name: str
        Name of the strategy
    results: dict
        Dictionary containing strategy backtest results
    directory: str
        Directory to save the results to
        
    Returns:
    --------
    str
        Path to the saved file
    """
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Create a copy of the results that can be serialized
    serializable_results = {}
    
    # Convert pandas Series/DataFrames to lists
    for key, value in results.items():
        if isinstance(value, pd.Series) or isinstance(value, pd.DataFrame):
            serializable_results[key] = value.to_dict()
        elif isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, np.integer):
            serializable_results[key] = int(value)
        elif isinstance(value, np.floating):
            serializable_results[key] = float(value)
        elif isinstance(value, list) and value and isinstance(value[0], pd.Timestamp):
            serializable_results[key] = [str(item) for item in value]
        else:
            serializable_results[key] = value
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{strategy_name}_{timestamp}.json"
    filepath = os.path.join(directory, filename)
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    return filepath

def load_strategy_results(filepath):
    """
    Load strategy backtest results from a JSON file
    
    Parameters:
    -----------
    filepath: str
        Path to the JSON file
        
    Returns:
    --------
    dict
        Dictionary containing strategy backtest results
    """
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    # Convert back to pandas objects if needed
    if 'equity_curve' in results:
        results['equity_curve'] = pd.Series(results['equity_curve'])
    if 'drawdowns' in results:
        results['drawdowns'] = pd.Series(results['drawdowns'])
    if 'returns' in results:
        results['returns'] = pd.Series(results['returns'])
    if 'monthly_returns' in results:
        results['monthly_returns'] = pd.DataFrame(results['monthly_returns'])
    
    # Convert buy/sell signals back to datetime if needed
    if 'buy_signals' in results and results['buy_signals'] and isinstance(results['buy_signals'][0], str):
        results['buy_signals'] = [pd.Timestamp(ts) for ts in results['buy_signals']]
    if 'sell_signals' in results and results['sell_signals'] and isinstance(results['sell_signals'][0], str):
        results['sell_signals'] = [pd.Timestamp(ts) for ts in results['sell_signals']]
    
    return results

def create_monthly_returns_heatmap(returns_series):
    """
    Create a monthly returns heatmap from a returns series
    
    Parameters:
    -----------
    returns_series: pandas.Series
        Series containing return values with DatetimeIndex
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with monthly returns (rows: years, columns: months)
    """
    if not isinstance(returns_series.index, pd.DatetimeIndex):
        return pd.DataFrame()
    
    # Resample to monthly returns
    monthly_returns = returns_series.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # Create a pivot table
    monthly_pivot = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })
    
    # Convert to pivot
    monthly_heatmap = monthly_pivot.pivot(index='Year', columns='Month', values='Return')
    
    # Convert to percentages
    monthly_heatmap = monthly_heatmap * 100
    
    return monthly_heatmap

def parse_timeframe(timeframe_str):
    """
    Parse a timeframe string into minutes
    
    Parameters:
    -----------
    timeframe_str: str
        Timeframe string (e.g., '1m', '5m', '1h', '1d')
        
    Returns:
    --------
    int
        Number of minutes for the timeframe
    """
    if not timeframe_str:
        return 1440  # Default to daily (1440 minutes)
    
    # Extract number and unit
    if timeframe_str[-1].isdigit():
        return int(timeframe_str)  # Already in minutes
    
    amount = int(timeframe_str[:-1])
    unit = timeframe_str[-1].lower()
    
    if unit == 'm':
        return amount
    elif unit == 'h':
        return amount * 60
    elif unit == 'd':
        return amount * 1440
    elif unit == 'w':
        return amount * 1440 * 7
    else:
        return 1440  # Default to daily
