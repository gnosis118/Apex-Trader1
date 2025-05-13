import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

# Get database URL from environment variable
DATABASE_URL = os.environ.get("DATABASE_URL")

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create Base class for declarative models
Base = declarative_base()

# Define database models
class MarketData(Base):
    """Store historical market data for different symbols"""
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float)
    
    def __repr__(self):
        return f"<MarketData(symbol='{self.symbol}', date='{self.date}', close={self.close})>"

class Strategy(Base):
    """Store trading strategy configurations"""
    __tablename__ = 'strategies'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    type = Column(String(50), nullable=False)  # e.g., "MA Crossover", "RSI", etc.
    description = Column(Text)
    parameters = Column(Text)  # JSON string of parameters
    creation_date = Column(DateTime, default=datetime.utcnow)
    
    # Define relationship with backtest results
    backtest_results = relationship("BacktestResult", back_populates="strategy")
    
    def __repr__(self):
        return f"<Strategy(name='{self.name}', type='{self.type}')>"

class BacktestResult(Base):
    """Store results of strategy backtests"""
    __tablename__ = 'backtest_results'
    
    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey('strategies.id'))
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    total_return = Column(Float)
    annualized_return = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    trades_count = Column(Integer)
    execution_date = Column(DateTime, default=datetime.utcnow)
    
    # Define relationship with strategy
    strategy = relationship("Strategy", back_populates="backtest_results")
    
    def __repr__(self):
        return f"<BacktestResult(strategy='{self.strategy.name}', symbol='{self.symbol}', total_return={self.total_return})>"

class Trade(Base):
    """Store individual trades from backtests or live trading"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    backtest_id = Column(Integer, ForeignKey('backtest_results.id'), nullable=True)
    symbol = Column(String(20), nullable=False)
    entry_date = Column(DateTime, nullable=False)
    exit_date = Column(DateTime, nullable=True)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    direction = Column(Integer, nullable=False)  # 1 for long, -1 for short
    size = Column(Float, nullable=False)
    profit_loss = Column(Float, nullable=True)
    status = Column(String(20), nullable=False)  # "open", "closed", "cancelled"
    
    def __repr__(self):
        return f"<Trade(symbol='{self.symbol}', direction={self.direction}, profit_loss={self.profit_loss})>"

# Create all tables
def initialize_database():
    """Create all database tables if they don't exist"""
    Base.metadata.create_all(engine)
    print("Database initialized successfully")

# Create a session factory
SessionFactory = sessionmaker(bind=engine)

# Create a database session
def get_session():
    """Get a new database session"""
    return SessionFactory()

# Functions to interact with market data
def save_market_data(df, symbol, timeframe):
    """
    Save a DataFrame of market data to the database
    
    Parameters:
    -----------
    df: pandas.DataFrame
        DataFrame with market data (must have columns: Date, Open, High, Low, Close, Volume)
    symbol: str
        The trading symbol (e.g., "ES=F")
    timeframe: str
        The data timeframe (e.g., "1d", "1h")
    """
    session = get_session()
    
    try:
        # Prepare data for insertion
        data_records = []
        for index, row in df.iterrows():
            record = MarketData(
                symbol=symbol,
                date=index if isinstance(index, datetime) else pd.to_datetime(index),
                timeframe=timeframe,
                open=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close'],
                volume=row['Volume'] if 'Volume' in row else None
            )
            data_records.append(record)
        
        # Bulk insert
        session.bulk_save_objects(data_records)
        session.commit()
        print(f"Saved {len(data_records)} records for {symbol} ({timeframe})")
        
    except Exception as e:
        session.rollback()
        print(f"Error saving market data: {str(e)}")
        raise
    finally:
        session.close()

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
    session = get_session()
    
    try:
        # Query data
        query = (
            session.query(MarketData)
            .filter(
                MarketData.symbol == symbol,
                MarketData.timeframe == timeframe,
                MarketData.date >= start_date,
                MarketData.date <= end_date
            )
            .order_by(MarketData.date)
        )
        
        # Convert to DataFrame
        records = []
        for record in query:
            records.append({
                'Date': record.date,
                'Open': record.open,
                'High': record.high,
                'Low': record.low,
                'Close': record.close,
                'Volume': record.volume
            })
        
        df = pd.DataFrame(records)
        
        # Set date as index if data exists
        if not df.empty:
            df.set_index('Date', inplace=True)
        
        return df
        
    except Exception as e:
        print(f"Error loading market data: {str(e)}")
        raise
    finally:
        session.close()

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
    session = get_session()
    
    try:
        # Convert parameters dict to JSON string
        import json
        params_json = json.dumps(parameters)
        
        # Create strategy record
        strategy = Strategy(
            name=name,
            type=strategy_type,
            description=description,
            parameters=params_json
        )
        
        # Save to database
        session.add(strategy)
        session.commit()
        
        return strategy.id
        
    except Exception as e:
        session.rollback()
        print(f"Error saving strategy: {str(e)}")
        raise
    finally:
        session.close()

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
    session = get_session()
    
    try:
        # Create backtest result record
        backtest = BacktestResult(
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            total_return=results.get('total_return'),
            annualized_return=results.get('annualized_return'),
            sharpe_ratio=results.get('sharpe_ratio'),
            max_drawdown=results.get('max_drawdown'),
            win_rate=results.get('win_rate'),
            profit_factor=results.get('profit_factor'),
            trades_count=results.get('trades_count', 0)
        )
        
        # Save to database
        session.add(backtest)
        session.commit()
        
        # Save individual trades if available
        trades = results.get('trades', [])
        for trade_data in trades:
            trade = Trade(
                backtest_id=backtest.id,
                symbol=symbol,
                entry_date=trade_data.get('entry_date'),
                exit_date=trade_data.get('exit_date'),
                entry_price=trade_data.get('entry_price'),
                exit_price=trade_data.get('exit_price'),
                direction=trade_data.get('direction'),
                size=trade_data.get('size', 1.0),
                profit_loss=trade_data.get('profit_pct'),
                status="closed"
            )
            session.add(trade)
        
        session.commit()
        
        return backtest.id
        
    except Exception as e:
        session.rollback()
        print(f"Error saving backtest result: {str(e)}")
        raise
    finally:
        session.close()

def get_strategies():
    """
    Get all strategies from the database
    
    Returns:
    --------
    list
        List of strategy dictionaries
    """
    session = get_session()
    
    try:
        strategies = []
        query = session.query(Strategy).order_by(Strategy.creation_date.desc())
        
        for strategy in query:
            import json
            parameters = json.loads(strategy.parameters) if strategy.parameters else {}
            
            strategies.append({
                'id': strategy.id,
                'name': strategy.name,
                'type': strategy.type,
                'description': strategy.description,
                'parameters': parameters,
                'creation_date': strategy.creation_date
            })
        
        return strategies
        
    except Exception as e:
        print(f"Error getting strategies: {str(e)}")
        raise
    finally:
        session.close()

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
    session = get_session()
    
    try:
        results = []
        query = session.query(BacktestResult).order_by(BacktestResult.execution_date.desc())
        
        if strategy_id is not None:
            query = query.filter(BacktestResult.strategy_id == strategy_id)
        
        for result in query:
            results.append({
                'id': result.id,
                'strategy_id': result.strategy_id,
                'strategy_name': result.strategy.name,
                'symbol': result.symbol,
                'timeframe': result.timeframe,
                'start_date': result.start_date,
                'end_date': result.end_date,
                'total_return': result.total_return,
                'annualized_return': result.annualized_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'trades_count': result.trades_count,
                'execution_date': result.execution_date
            })
        
        return results
        
    except Exception as e:
        print(f"Error getting backtest results: {str(e)}")
        raise
    finally:
        session.close()

# Check connection and initialize database
if __name__ == "__main__":
    try:
        initialize_database()
        print("Database connection successful")
    except Exception as e:
        print(f"Database connection failed: {str(e)}")