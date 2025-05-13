import pandas as pd
import numpy as np

class CustomTA:
    """
    Class for calculating technical indicators without external dependencies
    """
    
    def __init__(self, market_data):
        """
        Initialize the CustomTA class
        
        Parameters:
        -----------
        market_data: pandas.DataFrame
            DataFrame containing market data (OHLCV)
        """
        self.data = market_data.copy()
    
    def add_all_indicators(self):
        """
        Add all technical indicators to the data
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with all technical indicators added
        """
        # Add moving averages
        self.add_moving_averages([20, 50, 100, 200])
        
        # Add RSI
        self.add_rsi(14)
        
        # Add MACD
        self.add_macd()
        
        # Add Bollinger Bands
        self.add_bollinger_bands(20, 2.0)
        
        # Add ATR
        self.add_atr(14)
        
        # Add stochastic oscillator
        self.add_stochastic(14, 3)
        
        # Add volume indicators
        self.add_volume_indicators()
        
        return self.data
    
    def add_moving_averages(self, periods):
        """
        Add moving averages to the data
        
        Parameters:
        -----------
        periods: list
            List of periods for moving averages
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with moving averages added
        """
        for period in periods:
            self.data[f'MA_{period}'] = self.data['Close'].rolling(window=period).mean()
            self.data[f'EMA_{period}'] = self.data['Close'].ewm(span=period, adjust=False).mean()
        
        return self.data
    
    def add_rsi(self, period=14):
        """
        Add RSI indicator to the data
        
        Parameters:
        -----------
        period: int
            Period for RSI calculation
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with RSI added
        """
        delta = self.data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS and RSI for the rest of the data
        for i in range(period+1, len(self.data)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period-1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period-1) + loss.iloc[i]) / period
        
        rs = avg_gain / avg_loss
        self.data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        return self.data
    
    def add_macd(self, fast_period=12, slow_period=26, signal_period=9):
        """
        Add MACD indicator to the data
        
        Parameters:
        -----------
        fast_period: int
            Fast period for MACD calculation
        slow_period: int
            Slow period for MACD calculation
        signal_period: int
            Signal period for MACD calculation
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with MACD added
        """
        # Calculate fast and slow EMAs
        fast_ema = self.data['Close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = self.data['Close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        self.data['MACD'] = fast_ema - slow_ema
        
        # Calculate signal line
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        self.data['MACD_Hist'] = self.data['MACD'] - self.data['MACD_Signal']
        
        return self.data
    
    def add_bollinger_bands(self, period=20, std_dev=2.0):
        """
        Add Bollinger Bands to the data
        
        Parameters:
        -----------
        period: int
            Period for Bollinger Bands calculation
        std_dev: float
            Number of standard deviations for the bands
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with Bollinger Bands added
        """
        # Calculate middle band (SMA)
        self.data[f'BB_Middle_{period}'] = self.data['Close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        rolling_std = self.data['Close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        self.data[f'BB_Upper_{period}'] = self.data[f'BB_Middle_{period}'] + (rolling_std * std_dev)
        self.data[f'BB_Lower_{period}'] = self.data[f'BB_Middle_{period}'] - (rolling_std * std_dev)
        
        # Calculate Bollinger Bands Width
        self.data[f'BB_Width_{period}'] = (self.data[f'BB_Upper_{period}'] - self.data[f'BB_Lower_{period}']) / self.data[f'BB_Middle_{period}']
        
        return self.data
    
    def add_atr(self, period=14):
        """
        Add Average True Range (ATR) to the data
        
        Parameters:
        -----------
        period: int
            Period for ATR calculation
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with ATR added
        """
        # Calculate True Range
        tr1 = self.data['High'] - self.data['Low']
        tr2 = abs(self.data['High'] - self.data['Close'].shift())
        tr3 = abs(self.data['Low'] - self.data['Close'].shift())
        
        # Get the maximum of the three true ranges
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        
        # Calculate ATR as the simple moving average of true range
        self.data[f'ATR_{period}'] = tr.rolling(window=period).mean()
        
        return self.data
    
    def add_stochastic(self, k_period=14, d_period=3):
        """
        Add Stochastic Oscillator to the data
        
        Parameters:
        -----------
        k_period: int
            K period for Stochastic calculation
        d_period: int
            D period for Stochastic calculation
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with Stochastic Oscillator added
        """
        # Calculate %K
        lowest_low = self.data['Low'].rolling(window=k_period).min()
        highest_high = self.data['High'].rolling(window=k_period).max()
        
        # Handle division by zero
        denom = highest_high - lowest_low
        denom = denom.replace(0, 0.00001)
        
        self.data['Stoch_K'] = 100 * ((self.data['Close'] - lowest_low) / denom)
        
        # Calculate %D
        self.data['Stoch_D'] = self.data['Stoch_K'].rolling(window=d_period).mean()
        
        return self.data
    
    def add_volume_indicators(self):
        """
        Add volume-based indicators to the data
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with volume indicators added
        """
        # Calculate volume moving average
        self.data['Volume_MA_20'] = self.data['Volume'].rolling(window=20).mean()
        
        # Calculate volume change percentage
        self.data['Volume_Change'] = self.data['Volume'].pct_change() * 100
        
        # Calculate price-volume trend
        self.data['PVT'] = (self.data['Close'].pct_change() * self.data['Volume']).cumsum()
        
        return self.data
    
    def add_custom_indicators(self):
        """
        Add custom indicators specifically for futures markets
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with custom indicators added
        """
        # Calculate return series if not exists
        if 'Returns' not in self.data.columns:
            self.data['Returns'] = self.data['Close'].pct_change()
            
        # Calculate price momentum (rate of change)
        self.data['ROC_5'] = self.data['Close'].pct_change(periods=5) * 100
        self.data['ROC_10'] = self.data['Close'].pct_change(periods=10) * 100
        self.data['ROC_20'] = self.data['Close'].pct_change(periods=20) * 100
        
        # Calculate volatility ratio (current volatility / historical volatility)
        short_vol = self.data['Returns'].rolling(window=5).std() * np.sqrt(252)
        long_vol = self.data['Returns'].rolling(window=20).std() * np.sqrt(252)
        # Avoid division by zero
        long_vol = long_vol.replace(0, 0.00001)
        self.data['Volatility_Ratio'] = short_vol / long_vol
        
        # Volume pressure (volume * price direction)
        self.data['Volume_Pressure'] = self.data['Volume'] * np.sign(self.data['Close'] - self.data['Close'].shift(1))
        
        # Add custom indicators for trend strength
        if 'MA_50' in self.data.columns:
            # Avoid division by zero
            denom = self.data['MA_50'].replace(0, 0.00001) 
            self.data['Trend_Strength'] = np.abs(self.data['Close'] - self.data['MA_50']) / denom
        
        # Futures specific - price vs moving average divergence
        if 'MA_20' in self.data.columns and 'ATR_14' in self.data.columns:
            # Avoid division by zero
            atr_safe = self.data['ATR_14'].replace(0, 0.00001)
            self.data['MA_Divergence'] = (self.data['Close'] - self.data['MA_20']) / atr_safe
        
        return self.data