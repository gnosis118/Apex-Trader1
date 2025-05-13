import pandas as pd
import numpy as np
import talib as ta

class TechnicalAnalysis:
    """
    Class for calculating technical indicators
    """
    
    def __init__(self, market_data):
        """
        Initialize the TechnicalAnalysis class
        
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
        
        # Add Stochastic oscillator
        self.add_stochastic(14, 3)
        
        # Add ADX
        self.add_adx(14)
        
        # Add CCI
        self.add_cci(20)
        
        # Add Williams %R
        self.add_williams_r(14)
        
        # Add OBV
        self.add_obv()
        
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
            self.data[f'MA_{period}'] = ta.SMA(self.data['Close'].values, timeperiod=period)
            self.data[f'EMA_{period}'] = ta.EMA(self.data['Close'].values, timeperiod=period)
        
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
        self.data[f'RSI_{period}'] = ta.RSI(self.data['Close'].values, timeperiod=period)
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
        macd, macd_signal, macd_hist = ta.MACD(
            self.data['Close'].values,
            fastperiod=fast_period,
            slowperiod=slow_period,
            signalperiod=signal_period
        )
        
        self.data['MACD'] = macd
        self.data['MACD_Signal'] = macd_signal
        self.data['MACD_Hist'] = macd_hist
        
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
        upper, middle, lower = ta.BBANDS(
            self.data['Close'].values, 
            timeperiod=period,
            nbdevup=std_dev,
            nbdevdn=std_dev,
            matype=0
        )
        
        self.data[f'BB_Upper_{period}'] = upper
        self.data[f'BB_Middle_{period}'] = middle
        self.data[f'BB_Lower_{period}'] = lower
        
        # Calculate Bollinger Bands Width
        self.data[f'BB_Width_{period}'] = (upper - lower) / middle
        
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
        self.data[f'ATR_{period}'] = ta.ATR(
            self.data['High'].values,
            self.data['Low'].values,
            self.data['Close'].values,
            timeperiod=period
        )
        
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
        slowk, slowd = ta.STOCH(
            self.data['High'].values,
            self.data['Low'].values,
            self.data['Close'].values,
            fastk_period=k_period,
            slowk_period=d_period,
            slowk_matype=0,
            slowd_period=d_period,
            slowd_matype=0
        )
        
        self.data['Stoch_K'] = slowk
        self.data['Stoch_D'] = slowd
        
        return self.data
    
    def add_adx(self, period=14):
        """
        Add Average Directional Index (ADX) to the data
        
        Parameters:
        -----------
        period: int
            Period for ADX calculation
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with ADX added
        """
        self.data['ADX'] = ta.ADX(
            self.data['High'].values,
            self.data['Low'].values,
            self.data['Close'].values,
            timeperiod=period
        )
        
        # Also add DI+ and DI-
        self.data['DI+'] = ta.PLUS_DI(
            self.data['High'].values,
            self.data['Low'].values,
            self.data['Close'].values,
            timeperiod=period
        )
        
        self.data['DI-'] = ta.MINUS_DI(
            self.data['High'].values,
            self.data['Low'].values,
            self.data['Close'].values,
            timeperiod=period
        )
        
        return self.data
    
    def add_cci(self, period=20):
        """
        Add Commodity Channel Index (CCI) to the data
        
        Parameters:
        -----------
        period: int
            Period for CCI calculation
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with CCI added
        """
        self.data['CCI'] = ta.CCI(
            self.data['High'].values,
            self.data['Low'].values,
            self.data['Close'].values,
            timeperiod=period
        )
        
        return self.data
    
    def add_williams_r(self, period=14):
        """
        Add Williams %R to the data
        
        Parameters:
        -----------
        period: int
            Period for Williams %R calculation
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with Williams %R added
        """
        self.data['Williams_%R'] = ta.WILLR(
            self.data['High'].values,
            self.data['Low'].values,
            self.data['Close'].values,
            timeperiod=period
        )
        
        return self.data
    
    def add_obv(self):
        """
        Add On-Balance Volume (OBV) to the data
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with OBV added
        """
        self.data['OBV'] = ta.OBV(
            self.data['Close'].values,
            self.data['Volume'].values
        )
        
        return self.data
    
    def add_custom_indicators(self):
        """
        Add custom indicators specifically for futures markets
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with custom indicators added
        """
        # Calculate price momentum (rate of change)
        self.data['ROC_5'] = ta.ROC(self.data['Close'].values, timeperiod=5)
        self.data['ROC_10'] = ta.ROC(self.data['Close'].values, timeperiod=10)
        self.data['ROC_20'] = ta.ROC(self.data['Close'].values, timeperiod=20)
        
        # Calculate volatility ratio (current volatility / historical volatility)
        short_vol = self.data['Returns'].rolling(window=5).std() * np.sqrt(252)
        long_vol = self.data['Returns'].rolling(window=20).std() * np.sqrt(252)
        self.data['Volatility_Ratio'] = short_vol / long_vol
        
        # Volume pressure (volume * price direction)
        self.data['Volume_Pressure'] = self.data['Volume'] * np.sign(self.data['Close'] - self.data['Close'].shift(1))
        
        # Add custom indicators for trend strength
        self.data['Trend_Strength'] = np.abs(self.data['Close'] - self.data['MA_50']) / self.data['MA_50']
        
        # Futures specific - price vs moving average divergence
        self.data['MA_Divergence'] = (self.data['Close'] - self.data['MA_20']) / self.data['ATR_14']
        
        return self.data
