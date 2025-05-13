import json
import os
import pandas as pd
from datetime import datetime

# Directory for storing data files
DATA_DIR = "data"
MARKET_DATA_DIR = os.path.join(DATA_DIR, "market_data")
STRATEGIES_DIR = os.path.join(DATA_DIR, "strategies")
BACKTEST_DIR = os.path.join(DATA_DIR, "backtests")

# Ensure directories exist
def initialize_storage():
    """Create directories for data storage"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MARKET_DATA_DIR, exist_ok=True)
    os.makedirs(STRATEGIES_DIR, exist_ok=True)
    os.makedirs(BACKTEST_DIR, exist_ok=True)
    print("File storage initialized")

def save_market_data(df, symbol, timeframe):
    """
    Save market data to CSV file
    
    Parameters:
    -----------
    df: pandas.DataFrame
        DataFrame with market data
    symbol: str
        Trading symbol (e.g., "ES=F")
    timeframe: str
        Timeframe (e.g., "1d", "1h")
    """
    initialize_storage()
    
    # Create filename
    filename = f"{symbol}_{timeframe}.csv"
    filepath = os.path.join(MARKET_DATA_DIR, filename)
    
    # Save DataFrame to CSV
    df.to_csv(filepath)
    print(f"Saved {len(df)} records for {symbol} ({timeframe}) to {filepath}")

def load_market_data(symbol, timeframe, start_date, end_date):
    """
    Load market data from CSV file
    
    Parameters:
    -----------
    symbol: str
        Trading symbol (e.g., "ES=F")
    timeframe: str
        Timeframe (e.g., "1d", "1h")
    start_date: datetime
        Start date for filtering data
    end_date: datetime
        End date for filtering data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with market data
    """
    initialize_storage()
    
    # Create filename
    filename = f"{symbol}_{timeframe}.csv"
    filepath = os.path.join(MARKET_DATA_DIR, filename)
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"No data found for {symbol} ({timeframe})")
        return pd.DataFrame()
    
    # Load data from CSV
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # Filter by date range
    if not df.empty:
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        # Filter by date range
        df = df[(df.index >= start_date) & (df.index <= end_date)]
    
    return df

def save_strategy(name, strategy_type, description, parameters):
    """
    Save strategy to JSON file
    
    Parameters:
    -----------
    name: str
        Strategy name
    strategy_type: str
        Type of strategy (e.g., "MA Crossover")
    description: str
        Description of the strategy
    parameters: dict
        Dictionary of strategy parameters
        
    Returns:
    --------
    int
        ID of the saved strategy
    """
    initialize_storage()
    
    # Create strategy object
    strategy = {
        "id": _get_next_id(STRATEGIES_DIR),
        "name": name,
        "type": strategy_type,
        "description": description,
        "parameters": parameters,
        "creation_date": datetime.now().isoformat()
    }
    
    # Save to JSON file
    filename = f"strategy_{strategy['id']}.json"
    filepath = os.path.join(STRATEGIES_DIR, filename)
    
    with open(filepath, 'w') as f:
        json.dump(strategy, f, indent=2)
        
    print(f"Saved strategy {name} (ID: {strategy['id']}) to {filepath}")
    return strategy['id']

def get_strategies():
    """
    Get all strategies
    
    Returns:
    --------
    list
        List of strategy dictionaries
    """
    initialize_storage()
    
    strategies = []
    
    # Load all strategy files
    for filename in os.listdir(STRATEGIES_DIR):
        if filename.startswith('strategy_') and filename.endswith('.json'):
            filepath = os.path.join(STRATEGIES_DIR, filename)
            
            with open(filepath, 'r') as f:
                strategy = json.load(f)
                strategies.append(strategy)
    
    # Sort by creation date (newest first)
    strategies.sort(key=lambda x: x.get('creation_date', ''), reverse=True)
    
    return strategies

def save_backtest_result(strategy_id, symbol, timeframe, start_date, end_date, results):
    """
    Save backtest results to JSON file
    
    Parameters:
    -----------
    strategy_id: int
        ID of the strategy used
    symbol: str
        Trading symbol
    timeframe: str
        Timeframe
    start_date: datetime
        Start date of backtest
    end_date: datetime
        End date of backtest
    results: dict
        Dictionary of backtest results
        
    Returns:
    --------
    int
        ID of the saved backtest result
    """
    initialize_storage()
    
    # Create backtest object
    backtest = {
        "id": _get_next_id(BACKTEST_DIR),
        "strategy_id": strategy_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "total_return": results.get('total_return'),
        "annualized_return": results.get('annualized_return'),
        "sharpe_ratio": results.get('sharpe_ratio'),
        "max_drawdown": results.get('max_drawdown'),
        "win_rate": results.get('win_rate'),
        "profit_factor": results.get('profit_factor'),
        "trades_count": results.get('trades_count', 0),
        "execution_date": datetime.now().isoformat()
    }
    
    # Save trades separately if they exist
    trades = results.get('trades', [])
    if trades:
        trades_data = []
        for trade in trades:
            # Convert any datetime objects to strings
            trade_copy = {}
            for key, value in trade.items():
                if isinstance(value, datetime):
                    trade_copy[key] = value.isoformat()
                else:
                    trade_copy[key] = value
            
            trade_copy["backtest_id"] = backtest["id"]
            trades_data.append(trade_copy)
        
        backtest["trades"] = trades_data
    
    # Find strategy name
    strategy_filename = f"strategy_{strategy_id}.json"
    strategy_filepath = os.path.join(STRATEGIES_DIR, strategy_filename)
    
    if os.path.exists(strategy_filepath):
        with open(strategy_filepath, 'r') as f:
            strategy = json.load(f)
            backtest["strategy_name"] = strategy.get("name", "Unknown Strategy")
    else:
        backtest["strategy_name"] = "Unknown Strategy"
    
    # Save to JSON file
    filename = f"backtest_{backtest['id']}.json"
    filepath = os.path.join(BACKTEST_DIR, filename)
    
    with open(filepath, 'w') as f:
        json.dump(backtest, f, indent=2)
        
    print(f"Saved backtest result (ID: {backtest['id']}) to {filepath}")
    return backtest['id']

def get_backtest_results(strategy_id=None):
    """
    Get backtest results
    
    Parameters:
    -----------
    strategy_id: int, optional
        Filter by strategy ID
        
    Returns:
    --------
    list
        List of backtest result dictionaries
    """
    initialize_storage()
    
    backtest_results = []
    
    # Load all backtest files
    for filename in os.listdir(BACKTEST_DIR):
        if filename.startswith('backtest_') and filename.endswith('.json'):
            filepath = os.path.join(BACKTEST_DIR, filename)
            
            with open(filepath, 'r') as f:
                backtest = json.load(f)
                
                # Filter by strategy_id if provided
                if strategy_id is None or backtest.get('strategy_id') == strategy_id:
                    backtest_results.append(backtest)
    
    # Sort by execution date (newest first)
    backtest_results.sort(key=lambda x: x.get('execution_date', ''), reverse=True)
    
    return backtest_results

def _get_next_id(directory):
    """
    Get the next available ID for a directory
    
    Parameters:
    -----------
    directory: str
        Directory path
        
    Returns:
    --------
    int
        Next available ID
    """
    # Get all files in directory
    files = os.listdir(directory)
    
    # Extract IDs from filenames
    ids = []
    for filename in files:
        if '_' in filename and filename.endswith('.json'):
            try:
                file_id = int(filename.split('_')[1].split('.')[0])
                ids.append(file_id)
            except (ValueError, IndexError):
                continue
    
    # Return next ID (max + 1, or 1 if no files exist)
    return max(ids) + 1 if ids else 1

def check_connection():
    """Check if file storage is working"""
    try:
        initialize_storage()
        return True
    except Exception as e:
        print(f"File storage error: {str(e)}")
        return False

# Initialize storage if this script is run directly
if __name__ == "__main__":
    initialize_storage()
    print("File storage system ready")