import os
import json
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime

# Get database connection parameters from environment variables
DB_HOST = os.environ.get("PGHOST")
DB_PORT = os.environ.get("PGPORT")
DB_NAME = os.environ.get("PGDATABASE")
DB_USER = os.environ.get("PGUSER")
DB_PASSWORD = os.environ.get("PGPASSWORD")

def get_connection():
    """Get a connection to the PostgreSQL database"""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

def initialize_database():
    """Create database tables if they don't exist"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Create market_data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                date TIMESTAMP NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                open FLOAT NOT NULL,
                high FLOAT NOT NULL,
                low FLOAT NOT NULL,
                close FLOAT NOT NULL,
                volume FLOAT,
                UNIQUE(symbol, date, timeframe)
            )
        """)
        
        # Create index on symbol and date
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_market_data_symbol_date 
            ON market_data(symbol, date)
        """)
        
        # Create strategies table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                type VARCHAR(50) NOT NULL,
                description TEXT,
                parameters TEXT,
                creation_date TIMESTAMP DEFAULT NOW()
            )
        """)
        
        # Create backtest_results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_results (
                id SERIAL PRIMARY KEY,
                strategy_id INTEGER REFERENCES strategies(id),
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                start_date TIMESTAMP NOT NULL,
                end_date TIMESTAMP NOT NULL,
                total_return FLOAT,
                annualized_return FLOAT,
                sharpe_ratio FLOAT,
                max_drawdown FLOAT,
                win_rate FLOAT,
                profit_factor FLOAT,
                trades_count INTEGER,
                execution_date TIMESTAMP DEFAULT NOW()
            )
        """)
        
        # Create trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                backtest_id INTEGER REFERENCES backtest_results(id),
                symbol VARCHAR(20) NOT NULL,
                entry_date TIMESTAMP NOT NULL,
                exit_date TIMESTAMP,
                entry_price FLOAT NOT NULL,
                exit_price FLOAT,
                direction INTEGER NOT NULL,
                size FLOAT NOT NULL,
                profit_loss FLOAT,
                status VARCHAR(20) NOT NULL
            )
        """)
        
        conn.commit()
        print("Database initialized successfully")
    except Exception as e:
        conn.rollback()
        print(f"Error initializing database: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def save_market_data(df, symbol, timeframe):
    """
    Save a DataFrame of market data to the database
    
    Parameters:
    -----------
    df: pandas.DataFrame
        DataFrame with market data (must have columns: Open, High, Low, Close, Volume)
    symbol: str
        The trading symbol (e.g., "ES=F")
    timeframe: str
        The data timeframe (e.g., "1d", "1h")
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Prepare data for insertion
        data_records = []
        for index, row in df.iterrows():
            date = index if isinstance(index, datetime) else pd.to_datetime(index)
            record = (
                symbol,
                date,
                timeframe,
                row['Open'],
                row['High'],
                row['Low'],
                row['Close'],
                row['Volume'] if 'Volume' in row else None
            )
            data_records.append(record)
        
        # Use COPY for faster insertion
        args_str = ','.join(cursor.mogrify("(%s,%s,%s,%s,%s,%s,%s,%s)", x).decode('utf-8') for x in data_records)
        
        if args_str:
            cursor.execute(f"""
                INSERT INTO market_data (symbol, date, timeframe, open, high, low, close, volume)
                VALUES {args_str}
                ON CONFLICT (symbol, date, timeframe) DO UPDATE
                SET open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
            """)
        
        conn.commit()
        print(f"Saved {len(data_records)} records for {symbol} ({timeframe})")
        
    except Exception as e:
        conn.rollback()
        print(f"Error saving market data: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def load_market_data(symbol, timeframe, start_date, end_date):
    """
    Load market data from the database
    
    Parameters:
    -----------
    symbol: str
        The trading symbol (e.g., "ES=F")
    timeframe: str
        The data timeframe (e.g., "1d", "1h")
    start_date: datetime
        The start date for the data
    end_date: datetime
        The end date for the data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with market data
    """
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Query data
        cursor.execute("""
            SELECT * FROM market_data
            WHERE symbol = %s AND timeframe = %s
            AND date BETWEEN %s AND %s
            ORDER BY date
        """, (symbol, timeframe, start_date, end_date))
        
        records = cursor.fetchall()
        
        # Convert to DataFrame
        if records:
            df = pd.DataFrame(records)
            
            # Set date as index if data exists
            if not df.empty:
                df = df.rename(columns={
                    'date': 'Date',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                df.set_index('Date', inplace=True)
            
            return df
        else:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        
    except Exception as e:
        print(f"Error loading market data: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def save_strategy(name, strategy_type, description, parameters):
    """
    Save a trading strategy to the database
    
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
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Convert parameters dict to JSON string
        params_json = json.dumps(parameters)
        
        # Insert strategy
        cursor.execute("""
            INSERT INTO strategies (name, type, description, parameters)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """, (name, strategy_type, description, params_json))
        
        strategy_id = cursor.fetchone()[0]
        conn.commit()
        
        return strategy_id
        
    except Exception as e:
        conn.rollback()
        print(f"Error saving strategy: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def save_backtest_result(strategy_id, symbol, timeframe, start_date, end_date, results):
    """
    Save backtest results to the database
    
    Parameters:
    -----------
    strategy_id: int
        ID of the strategy used
    symbol: str
        The trading symbol
    timeframe: str
        The data timeframe
    start_date: datetime
        Start date of the backtest
    end_date: datetime
        End date of the backtest
    results: dict
        Dictionary of backtest results
        
    Returns:
    --------
    int
        ID of the saved backtest result
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Insert backtest result
        cursor.execute("""
            INSERT INTO backtest_results 
            (strategy_id, symbol, timeframe, start_date, end_date, 
             total_return, annualized_return, sharpe_ratio, max_drawdown, 
             win_rate, profit_factor, trades_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            strategy_id, symbol, timeframe, start_date, end_date,
            results.get('total_return'),
            results.get('annualized_return'),
            results.get('sharpe_ratio'),
            results.get('max_drawdown'),
            results.get('win_rate'),
            results.get('profit_factor'),
            results.get('trades_count', 0)
        ))
        
        backtest_id = cursor.fetchone()[0]
        
        # Save individual trades if available
        trades = results.get('trades', [])
        for trade_data in trades:
            cursor.execute("""
                INSERT INTO trades
                (backtest_id, symbol, entry_date, exit_date, entry_price, exit_price, 
                 direction, size, profit_loss, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                backtest_id, symbol,
                trade_data.get('entry_date'),
                trade_data.get('exit_date'),
                trade_data.get('entry_price'),
                trade_data.get('exit_price'),
                trade_data.get('direction'),
                trade_data.get('size', 1.0),
                trade_data.get('profit_pct'),
                "closed"
            ))
        
        conn.commit()
        return backtest_id
        
    except Exception as e:
        conn.rollback()
        print(f"Error saving backtest result: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def get_strategies():
    """
    Get all strategies from the database
    
    Returns:
    --------
    list
        List of strategy dictionaries
    """
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        cursor.execute("""
            SELECT * FROM strategies
            ORDER BY creation_date DESC
        """)
        
        strategies = cursor.fetchall()
        
        # Parse JSON parameters
        for strategy in strategies:
            if strategy['parameters']:
                strategy['parameters'] = json.loads(strategy['parameters'])
            else:
                strategy['parameters'] = {}
        
        return strategies
        
    except Exception as e:
        print(f"Error getting strategies: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def get_backtest_results(strategy_id=None):
    """
    Get backtest results from the database
    
    Parameters:
    -----------
    strategy_id: int, optional
        Filter by strategy ID
        
    Returns:
    --------
    list
        List of backtest result dictionaries
    """
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        if strategy_id is not None:
            cursor.execute("""
                SELECT br.*, s.name as strategy_name
                FROM backtest_results br
                JOIN strategies s ON br.strategy_id = s.id
                WHERE br.strategy_id = %s
                ORDER BY br.execution_date DESC
            """, (strategy_id,))
        else:
            cursor.execute("""
                SELECT br.*, s.name as strategy_name
                FROM backtest_results br
                JOIN strategies s ON br.strategy_id = s.id
                ORDER BY br.execution_date DESC
            """)
        
        return cursor.fetchall()
        
    except Exception as e:
        print(f"Error getting backtest results: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

# Check connection function
def check_connection():
    """Check if database connection is working"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        return False

# Initialize database if this script is run directly
if __name__ == "__main__":
    if check_connection():
        initialize_database()
        print("Database connection successful")
    else:
        print("Failed to connect to database")