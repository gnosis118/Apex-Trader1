import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

class CustomDataProcessor:
    """
    Class for downloading and processing market data without external dependencies
    """
    
    def __init__(self, symbol, timeframe, start_date, end_date):
        """
        Initialize the CustomDataProcessor
        
        Parameters:
        -----------
        symbol: str
            The symbol of the futures contract to download
        timeframe: str
            The timeframe for the data (1m, 5m, 15m, 30m, 1h, 4h, 1d)
        start_date: datetime
            The start date for the data
        end_date: datetime
            The end date for the data
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        
    def get_data(self):
        """
        Download and process the market data
        
        Returns:
        --------
        pandas.DataFrame
            The processed market data
        """
        # Convert timeframe to yfinance interval format
        interval_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        
        # Adjust start date for intraday data (yfinance only provides 7 days of 1-min data)
        adjusted_start = self.start_date
        if self.timeframe == '1m' and (self.end_date - self.start_date).days > 7:
            adjusted_start = self.end_date - timedelta(days=7)
            print(f"Adjusted start date to {adjusted_start} due to yfinance 1-min data limitations")
        elif self.timeframe in ['5m', '15m', '30m'] and (self.end_date - self.start_date).days > 60:
            adjusted_start = self.end_date - timedelta(days=60)
            print(f"Adjusted start date to {adjusted_start} due to yfinance intraday data limitations")
        
        # Download data from Yahoo Finance
        try:
            self.data = yf.download(
                self.symbol,
                start=adjusted_start,
                end=self.end_date,
                interval=interval_map.get(self.timeframe, '1d'),
                progress=False
            )
        except Exception as e:
            raise Exception(f"Error downloading data: {str(e)}")
        
        # Check if data is empty
        if self.data.empty:
            raise Exception(f"No data found for {self.symbol} with timeframe {self.timeframe}")
        
        # Clean and process the data
        self.data = self._process_data(self.data)
        
        return self.data
    
    def _process_data(self, data):
        """
        Clean and process the downloaded data
        
        Parameters:
        -----------
        data: pandas.DataFrame
            The raw data to process
            
        Returns:
        --------
        pandas.DataFrame
            The processed data
        """
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Rename columns to standard format
        data = data.rename(columns={
            'Date': 'Date',
            'Datetime': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Adj Close': 'Adj_Close',
            'Volume': 'Volume'
        })
        
        # Remove rows with NaN values
        data = data.dropna()
        
        # Calculate additional basic features
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Calculate simple volatility (rolling standard deviation)
        data['Volatility'] = data['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Set date as index
        data = data.set_index('Date')
        
        return data
    
    def resample_data(self, new_timeframe):
        """
        Resample the data to a new timeframe
        
        Parameters:
        -----------
        new_timeframe: str
            The new timeframe to resample to (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            
        Returns:
        --------
        pandas.DataFrame
            The resampled data
        """
        if self.data is None:
            raise Exception("No data to resample. Call get_data() first.")
        
        # Convert timeframe to pandas resample rule
        resample_map = {
            '1m': '1Min',
            '5m': '5Min',
            '15m': '15Min',
            '30m': '30Min',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }
        
        rule = resample_map.get(new_timeframe)
        if not rule:
            raise ValueError(f"Invalid timeframe: {new_timeframe}")
        
        # Make sure Date is a datetime index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)
        
        # Resample OHLCV data
        resampled = self.data.resample(rule).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        
        # Drop rows with NaN values
        resampled = resampled.dropna()
        
        # Recalculate derived columns
        resampled['Returns'] = resampled['Close'].pct_change()
        resampled['Log_Returns'] = np.log(resampled['Close'] / resampled['Close'].shift(1))
        resampled['Volatility'] = resampled['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        return resampled