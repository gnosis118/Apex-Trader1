import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

class Backtester:
    """
    Class for backtesting trading strategies
    """
    
    def __init__(self, market_data):
        """
        Initialize the Backtester class
        
        Parameters:
        -----------
        market_data: pandas.DataFrame
            DataFrame containing market data with technical indicators
        """
        self.data = market_data.copy()
        
    def run_ma_crossover_strategy(self, fast_period=20, slow_period=50, position_size=0.1, 
                                 stop_loss=0.05, take_profit=0.1, max_drawdown=0.25):
        """
        Run moving average crossover strategy backtest
        
        Parameters:
        -----------
        fast_period: int
            Period for fast moving average
        slow_period: int
            Period for slow moving average
        position_size: float
            Position size as a fraction of portfolio value (0.0 to 1.0)
        stop_loss: float
            Stop loss percentage (0.0 to 1.0)
        take_profit: float
            Take profit percentage (0.0 to 1.0)
        max_drawdown: float
            Maximum allowed drawdown percentage (0.0 to 1.0)
            
        Returns:
        --------
        dict
            Dictionary containing backtest results
        """
        # Make sure the required columns exist
        fast_ma_col = f'MA_{fast_period}'
        slow_ma_col = f'MA_{slow_period}'
        
        if fast_ma_col not in self.data.columns:
            self.data[fast_ma_col] = self.data['Close'].rolling(window=fast_period).mean()
        
        if slow_ma_col not in self.data.columns:
            self.data[slow_ma_col] = self.data['Close'].rolling(window=slow_period).mean()
        
        # Create a copy of the data for strategy calculation
        df = self.data.copy()
        
        # Generate signals
        df['Signal'] = 0
        df.loc[df[fast_ma_col] > df[slow_ma_col], 'Signal'] = 1
        df.loc[df[fast_ma_col] < df[slow_ma_col], 'Signal'] = -1
        
        # Calculate daily positions (0 = no position, 1 = long, -1 = short)
        df['Position'] = df['Signal'].shift(1)
        df['Position'].fillna(0, inplace=True)
        
        # Calculate daily returns
        df['Strategy_Returns'] = df['Position'] * df['Returns']
        
        # Apply risk management
        df = self._apply_risk_management(df, position_size, stop_loss, take_profit, max_drawdown)
        
        # Calculate strategy metrics
        strategy_results = self._calculate_strategy_metrics(df)
        
        # Add trade analysis
        trades = self._extract_trades(df)
        strategy_results['trades'] = trades
        
        # Add buy/sell signals for visualization
        buy_signals = df.index[df['Position'].shift(-1) > df['Position']].tolist()
        sell_signals = df.index[df['Position'].shift(-1) < df['Position']].tolist()
        
        strategy_results['buy_signals'] = buy_signals
        strategy_results['sell_signals'] = sell_signals
        
        return strategy_results
    
    def run_rsi_strategy(self, rsi_period=14, overbought=70, oversold=30, position_size=0.1,
                        stop_loss=0.05, take_profit=0.1, max_drawdown=0.25):
        """
        Run RSI mean reversion strategy backtest
        
        Parameters:
        -----------
        rsi_period: int
            Period for RSI calculation
        overbought: float
            RSI level considered overbought
        oversold: float
            RSI level considered oversold
        position_size: float
            Position size as a fraction of portfolio value (0.0 to 1.0)
        stop_loss: float
            Stop loss percentage (0.0 to 1.0)
        take_profit: float
            Take profit percentage (0.0 to 1.0)
        max_drawdown: float
            Maximum allowed drawdown percentage (0.0 to 1.0)
            
        Returns:
        --------
        dict
            Dictionary containing backtest results
        """
        # Make sure the required columns exist
        rsi_col = f'RSI_{rsi_period}'
        
        if rsi_col not in self.data.columns:
            raise ValueError(f"RSI column {rsi_col} not found in data. Run add_rsi() first.")
        
        # Create a copy of the data for strategy calculation
        df = self.data.copy()
        
        # Generate signals
        df['Signal'] = 0
        df.loc[df[rsi_col] < oversold, 'Signal'] = 1  # Buy when RSI is oversold
        df.loc[df[rsi_col] > overbought, 'Signal'] = -1  # Sell when RSI is overbought
        
        # Calculate daily positions (0 = no position, 1 = long, -1 = short)
        # Only change position when signal changes
        df['Position'] = 0
        position = 0
        
        for i in range(1, len(df)):
            if df['Signal'].iloc[i] == 1 and position <= 0:
                position = 1  # Enter long position
            elif df['Signal'].iloc[i] == -1 and position >= 0:
                position = -1  # Enter short position
            
            df['Position'].iloc[i] = position
        
        # Calculate daily returns
        df['Strategy_Returns'] = df['Position'].shift(1) * df['Returns']
        df['Strategy_Returns'].fillna(0, inplace=True)
        
        # Apply risk management
        df = self._apply_risk_management(df, position_size, stop_loss, take_profit, max_drawdown)
        
        # Calculate strategy metrics
        strategy_results = self._calculate_strategy_metrics(df)
        
        # Add trade analysis
        trades = self._extract_trades(df)
        strategy_results['trades'] = trades
        
        # Add buy/sell signals for visualization
        buy_signals = df.index[(df['Position'] == 1) & (df['Position'].shift(1) != 1)].tolist()
        sell_signals = df.index[(df['Position'] == -1) & (df['Position'].shift(1) != -1)].tolist()
        
        strategy_results['buy_signals'] = buy_signals
        strategy_results['sell_signals'] = sell_signals
        
        return strategy_results
    
    def run_bollinger_bands_strategy(self, bb_period=20, bb_std=2.0, position_size=0.1,
                                   stop_loss=0.05, take_profit=0.1, max_drawdown=0.25):
        """
        Run Bollinger Bands strategy backtest
        
        Parameters:
        -----------
        bb_period: int
            Period for Bollinger Bands calculation
        bb_std: float
            Number of standard deviations for the bands
        position_size: float
            Position size as a fraction of portfolio value (0.0 to 1.0)
        stop_loss: float
            Stop loss percentage (0.0 to 1.0)
        take_profit: float
            Take profit percentage (0.0 to 1.0)
        max_drawdown: float
            Maximum allowed drawdown percentage (0.0 to 1.0)
            
        Returns:
        --------
        dict
            Dictionary containing backtest results
        """
        # Make sure the required columns exist
        upper_col = f'BB_Upper_{bb_period}'
        lower_col = f'BB_Lower_{bb_period}'
        
        if upper_col not in self.data.columns or lower_col not in self.data.columns:
            raise ValueError(f"Bollinger Bands columns not found in data. Run add_bollinger_bands() first.")
        
        # Create a copy of the data for strategy calculation
        df = self.data.copy()
        
        # Generate signals
        df['Signal'] = 0
        df.loc[df['Close'] < df[lower_col], 'Signal'] = 1  # Buy when price is below lower band
        df.loc[df['Close'] > df[upper_col], 'Signal'] = -1  # Sell when price is above upper band
        
        # Calculate daily positions (0 = no position, 1 = long, -1 = short)
        # Only change position when signal changes
        df['Position'] = 0
        position = 0
        
        for i in range(1, len(df)):
            if df['Signal'].iloc[i] == 1 and position <= 0:
                position = 1  # Enter long position
            elif df['Signal'].iloc[i] == -1 and position >= 0:
                position = -1  # Enter short position
            
            df['Position'].iloc[i] = position
        
        # Calculate daily returns
        df['Strategy_Returns'] = df['Position'].shift(1) * df['Returns']
        df['Strategy_Returns'].fillna(0, inplace=True)
        
        # Apply risk management
        df = self._apply_risk_management(df, position_size, stop_loss, take_profit, max_drawdown)
        
        # Calculate strategy metrics
        strategy_results = self._calculate_strategy_metrics(df)
        
        # Add trade analysis
        trades = self._extract_trades(df)
        strategy_results['trades'] = trades
        
        # Add buy/sell signals for visualization
        buy_signals = df.index[(df['Position'] == 1) & (df['Position'].shift(1) != 1)].tolist()
        sell_signals = df.index[(df['Position'] == -1) & (df['Position'].shift(1) != -1)].tolist()
        
        strategy_results['buy_signals'] = buy_signals
        strategy_results['sell_signals'] = sell_signals
        
        return strategy_results
    
    def run_macd_strategy(self, fast_period=12, slow_period=26, signal_period=9, position_size=0.1,
                         stop_loss=0.05, take_profit=0.1, max_drawdown=0.25):
        """
        Run MACD strategy backtest
        
        Parameters:
        -----------
        fast_period: int
            Fast period for MACD calculation
        slow_period: int
            Slow period for MACD calculation
        signal_period: int
            Signal period for MACD calculation
        position_size: float
            Position size as a fraction of portfolio value (0.0 to 1.0)
        stop_loss: float
            Stop loss percentage (0.0 to 1.0)
        take_profit: float
            Take profit percentage (0.0 to 1.0)
        max_drawdown: float
            Maximum allowed drawdown percentage (0.0 to 1.0)
            
        Returns:
        --------
        dict
            Dictionary containing backtest results
        """
        # Make sure the required columns exist
        if 'MACD' not in self.data.columns or 'MACD_Signal' not in self.data.columns:
            raise ValueError("MACD columns not found in data. Run add_macd() first.")
        
        # Create a copy of the data for strategy calculation
        df = self.data.copy()
        
        # Generate signals
        df['Signal'] = 0
        df.loc[df['MACD'] > df['MACD_Signal'], 'Signal'] = 1  # Buy when MACD crosses above signal line
        df.loc[df['MACD'] < df['MACD_Signal'], 'Signal'] = -1  # Sell when MACD crosses below signal line
        
        # Calculate daily positions (0 = no position, 1 = long, -1 = short)
        # Position changes when signal changes
        df['Position'] = df['Signal'].shift(1)
        df['Position'].fillna(0, inplace=True)
        
        # Calculate daily returns
        df['Strategy_Returns'] = df['Position'] * df['Returns']
        
        # Apply risk management
        df = self._apply_risk_management(df, position_size, stop_loss, take_profit, max_drawdown)
        
        # Calculate strategy metrics
        strategy_results = self._calculate_strategy_metrics(df)
        
        # Add trade analysis
        trades = self._extract_trades(df)
        strategy_results['trades'] = trades
        
        # Add buy/sell signals for visualization
        buy_signals = df.index[(df['Signal'] == 1) & (df['Signal'].shift(1) != 1)].tolist()
        sell_signals = df.index[(df['Signal'] == -1) & (df['Signal'].shift(1) != -1)].tolist()
        
        strategy_results['buy_signals'] = buy_signals
        strategy_results['sell_signals'] = sell_signals
        
        return strategy_results
    
    def run_combined_strategy(self, indicators, params, position_size=0.1,
                             stop_loss=0.05, take_profit=0.1, max_drawdown=0.25):
        """
        Run a combined strategy using multiple indicators
        
        Parameters:
        -----------
        indicators: list
            List of indicator names to use in the strategy (e.g., ['ma_crossover', 'rsi'])
        params: dict
            Dictionary of parameters for each indicator
        position_size: float
            Position size as a fraction of portfolio value (0.0 to 1.0)
        stop_loss: float
            Stop loss percentage (0.0 to 1.0)
        take_profit: float
            Take profit percentage (0.0 to 1.0)
        max_drawdown: float
            Maximum allowed drawdown percentage (0.0 to 1.0)
            
        Returns:
        --------
        dict
            Dictionary containing backtest results
        """
        # Create a copy of the data for strategy calculation
        df = self.data.copy()
        
        # Calculate signals for each indicator
        signal_columns = []
        
        for indicator in indicators:
            if indicator == 'ma_crossover':
                fast_ma = params.get('fast_ma', 20)
                slow_ma = params.get('slow_ma', 50)
                
                fast_ma_col = f'MA_{fast_ma}'
                slow_ma_col = f'MA_{slow_ma}'
                
                if fast_ma_col not in df.columns:
                    df[fast_ma_col] = df['Close'].rolling(window=fast_ma).mean()
                
                if slow_ma_col not in df.columns:
                    df[slow_ma_col] = df['Close'].rolling(window=slow_ma).mean()
                
                # Generate MA crossover signal
                signal_col = 'MA_Signal'
                df[signal_col] = 0
                df.loc[df[fast_ma_col] > df[slow_ma_col], signal_col] = 1
                df.loc[df[fast_ma_col] < df[slow_ma_col], signal_col] = -1
                
                signal_columns.append(signal_col)
            
            elif indicator == 'rsi':
                rsi_period = params.get('rsi_period', 14)
                rsi_overbought = params.get('rsi_overbought', 70)
                rsi_oversold = params.get('rsi_oversold', 30)
                
                rsi_col = f'RSI_{rsi_period}'
                
                if rsi_col not in df.columns:
                    raise ValueError(f"RSI column {rsi_col} not found in data.")
                
                # Generate RSI signal
                signal_col = 'RSI_Signal'
                df[signal_col] = 0
                df.loc[df[rsi_col] < rsi_oversold, signal_col] = 1
                df.loc[df[rsi_col] > rsi_overbought, signal_col] = -1
                
                signal_columns.append(signal_col)
            
            elif indicator == 'bollinger_bands':
                bb_period = params.get('bb_period', 20)
                bb_std = params.get('bb_std', 2.0)
                
                upper_col = f'BB_Upper_{bb_period}'
                lower_col = f'BB_Lower_{bb_period}'
                
                if upper_col not in df.columns or lower_col not in df.columns:
                    raise ValueError(f"Bollinger Bands columns not found in data.")
                
                # Generate Bollinger Bands signal
                signal_col = 'BB_Signal'
                df[signal_col] = 0
                df.loc[df['Close'] < df[lower_col], signal_col] = 1
                df.loc[df['Close'] > df[upper_col], signal_col] = -1
                
                signal_columns.append(signal_col)
            
            elif indicator == 'macd':
                if 'MACD' not in df.columns or 'MACD_Signal' not in df.columns:
                    raise ValueError("MACD columns not found in data.")
                
                # Generate MACD signal
                signal_col = 'MACD_Cross_Signal'
                df[signal_col] = 0
                df.loc[df['MACD'] > df['MACD_Signal'], signal_col] = 1
                df.loc[df['MACD'] < df['MACD_Signal'], signal_col] = -1
                
                signal_columns.append(signal_col)
        
        # Combine signals (simple average of signals)
        df['Combined_Signal'] = df[signal_columns].sum(axis=1) / len(signal_columns)
        
        # Convert combined signal to position (-1, 0, 1)
        df['Position'] = 0
        df.loc[df['Combined_Signal'] > 0.2, 'Position'] = 1
        df.loc[df['Combined_Signal'] < -0.2, 'Position'] = -1
        
        # Smooth positions to avoid excessive trading
        df['Position'] = df['Position'].rolling(window=3).mean().fillna(0)
        df['Position'] = np.where(df['Position'] > 0.5, 1, np.where(df['Position'] < -0.5, -1, 0))
        
        # Calculate daily returns
        df['Strategy_Returns'] = df['Position'].shift(1) * df['Returns']
        df['Strategy_Returns'].fillna(0, inplace=True)
        
        # Apply risk management
        df = self._apply_risk_management(df, position_size, stop_loss, take_profit, max_drawdown)
        
        # Calculate strategy metrics
        strategy_results = self._calculate_strategy_metrics(df)
        
        # Add trade analysis
        trades = self._extract_trades(df)
        strategy_results['trades'] = trades
        
        # Add buy/sell signals for visualization
        buy_signals = df.index[(df['Position'] == 1) & (df['Position'].shift(1) != 1)].tolist()
        sell_signals = df.index[(df['Position'] == -1) & (df['Position'].shift(1) != -1)].tolist()
        
        strategy_results['buy_signals'] = buy_signals
        strategy_results['sell_signals'] = sell_signals
        
        return strategy_results
    
    def _apply_risk_management(self, df, position_size, stop_loss, take_profit, max_drawdown):
        """
        Apply risk management rules to the strategy
        
        Parameters:
        -----------
        df: pandas.DataFrame
            DataFrame with strategy signals and positions
        position_size: float
            Position size as a fraction of portfolio value (0.0 to 1.0)
        stop_loss: float
            Stop loss percentage (0.0 to 1.0)
        take_profit: float
            Take profit percentage (0.0 to 1.0)
        max_drawdown: float
            Maximum allowed drawdown percentage (0.0 to 1.0)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with risk management applied
        """
        # Apply position sizing
        df['Strategy_Returns'] = df['Strategy_Returns'] * position_size
        
        # Initialize columns for tracking trade performance
        df['Entry_Price'] = np.nan
        df['Cumulative_Return'] = 0.0
        df['Trade_Return'] = 0.0
        df['Stop_Loss_Triggered'] = False
        df['Take_Profit_Triggered'] = False
        
        # Initialize equity and max equity
        equity = 1.0
        max_equity = 1.0
        drawdown = 0.0
        
        # Flag to track if position was closed due to risk management
        risk_closure = False
        
        # Process each day
        for i in range(1, len(df)):
            # Update equity
            if not risk_closure:
                equity *= (1 + df['Strategy_Returns'].iloc[i])
            else:
                risk_closure = False
            
            # Update max equity and drawdown
            max_equity = max(max_equity, equity)
            drawdown = (max_equity - equity) / max_equity
            
            # Check for position changes
            current_position = df['Position'].iloc[i]
            prev_position = df['Position'].iloc[i-1]
            
            # If entering a new position, record entry price
            if current_position != 0 and current_position != prev_position:
                df.loc[df.index[i], 'Entry_Price'] = df['Close'].iloc[i]
                df.loc[df.index[i], 'Trade_Return'] = 0.0
            
            # If in a position, calculate trade return
            if current_position != 0:
                entry_price = df['Entry_Price'].iloc[i] if not np.isnan(df['Entry_Price'].iloc[i]) else df['Entry_Price'].iloc[i-1]
                if not np.isnan(entry_price):
                    current_price = df['Close'].iloc[i]
                    trade_return = (current_price / entry_price - 1) * current_position
                    df.loc[df.index[i], 'Trade_Return'] = trade_return
                    
                    # Check stop loss
                    if trade_return < -stop_loss:
                        df.loc[df.index[i], 'Stop_Loss_Triggered'] = True
                        df.loc[df.index[i], 'Position'] = 0
                        risk_closure = True
                    
                    # Check take profit
                    if trade_return > take_profit:
                        df.loc[df.index[i], 'Take_Profit_Triggered'] = True
                        df.loc[df.index[i], 'Position'] = 0
                        risk_closure = True
            
            # Check max drawdown
            if drawdown > max_drawdown:
                df.loc[df.index[i:], 'Position'] = 0
                risk_closure = True
            
            # Update cumulative return
            df.loc[df.index[i], 'Cumulative_Return'] = equity - 1.0
        
        # Recalculate strategy returns after risk management
        df['Strategy_Returns_After_RM'] = df['Position'].shift(1) * df['Returns'] * position_size
        df['Strategy_Returns_After_RM'].fillna(0, inplace=True)
        
        # Calculate equity curve after risk management
        df['Equity_Curve'] = (1 + df['Strategy_Returns_After_RM']).cumprod()
        
        return df
    
    def _calculate_strategy_metrics(self, df):
        """
        Calculate performance metrics for the strategy
        
        Parameters:
        -----------
        df: pandas.DataFrame
            DataFrame with strategy results
            
        Returns:
        --------
        dict
            Dictionary with strategy performance metrics
        """
        # Use the risk-managed returns if available
        if 'Strategy_Returns_After_RM' in df.columns:
            returns_col = 'Strategy_Returns_After_RM'
        else:
            returns_col = 'Strategy_Returns'
        
        # Basic return metrics
        total_return = (df[returns_col] + 1).prod() - 1
        annual_return = (total_return + 1) ** (252 / len(df)) - 1
        
        # Risk metrics
        daily_returns = df[returns_col]
        volatility = daily_returns.std() * np.sqrt(252)
        downside_returns = daily_returns[daily_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Sharpe and Sortino ratios
        risk_free_rate = 0.02  # Assuming 2% risk-free rate
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max - 1)
        max_drawdown = drawdown.min() * 100  # Convert to percentage
        
        # Calculate Calmar ratio
        calmar_ratio = annual_return / (-max_drawdown / 100) if max_drawdown < 0 else 0
        
        # Trade metrics
        position_changes = df['Position'].diff().fillna(0)
        trades_started = position_changes != 0
        num_trades = trades_started.sum()
        
        # Calculate win rate and profit factor
        trades = self._extract_trades(df)
        if trades and len(trades) > 0:
            wins = sum(1 for trade in trades if trade['profit_pct'] > 0)
            losses = sum(1 for trade in trades if trade['profit_pct'] <= 0)
            
            win_rate = wins / len(trades) * 100 if len(trades) > 0 else 0
            
            total_profit = sum(trade['profit_pct'] for trade in trades if trade['profit_pct'] > 0)
            total_loss = sum(abs(trade['profit_pct']) for trade in trades if trade['profit_pct'] <= 0)
            
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            avg_win = total_profit / wins if wins > 0 else 0
            avg_loss = total_loss / losses if losses > 0 else 0
        else:
            win_rate = 0
            profit_factor = 0
            avg_win = 0
            avg_loss = 0
        
        # Risk metrics - VaR and Expected Shortfall
        var_95 = np.percentile(daily_returns, 5) * 100  # 95% VaR
        var_99 = np.percentile(daily_returns, 1) * 100  # 99% VaR
        
        es_95 = daily_returns[daily_returns <= np.percentile(daily_returns, 5)].mean() * 100 if len(daily_returns[daily_returns <= np.percentile(daily_returns, 5)]) > 0 else 0
        es_99 = daily_returns[daily_returns <= np.percentile(daily_returns, 1)].mean() * 100 if len(daily_returns[daily_returns <= np.percentile(daily_returns, 1)]) > 0 else 0
        
        # Monthly returns
        if isinstance(df.index, pd.DatetimeIndex):
            monthly_returns = df.groupby([df.index.year, df.index.month])[returns_col].apply(
                lambda x: (1 + x).prod() - 1
            ).reset_index()
            monthly_returns.columns = ['Year', 'Month', 'Return']
            
            # Convert to a pivot table
            monthly_returns_pivot = monthly_returns.pivot(index='Year', columns='Month', values='Return')
            
            # Convert to percentages
            monthly_returns_pivot = monthly_returns_pivot * 100
        else:
            monthly_returns_pivot = pd.DataFrame()
        
        # Create results dictionary
        results = {
            'total_return': total_return * 100,  # Convert to percentage
            'annual_return': annual_return * 100,  # Convert to percentage
            'volatility': volatility * 100,  # Convert to percentage
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': abs(max_drawdown),  # Already in percentage, take abs for display
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': int(num_trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'equity_curve': df['Equity_Curve'] if 'Equity_Curve' in df.columns else (1 + df[returns_col]).cumprod(),
            'drawdowns': drawdown * 100,  # Convert to percentage
            'returns': daily_returns,
            'var_95': var_95,
            'var_99': var_99,
            'es_95': es_95,
            'es_99': es_99
        }
        
        # Add monthly returns if available
        if not monthly_returns_pivot.empty:
            results['monthly_returns'] = monthly_returns_pivot
        
        return results
    
    def _extract_trades(self, df):
        """
        Extract individual trades from the backtest data
        
        Parameters:
        -----------
        df: pandas.DataFrame
            DataFrame with strategy results
            
        Returns:
        --------
        list
            List of trade dictionaries
        """
        trades = []
        
        # Find changes in position
        position_changes = df['Position'].diff().fillna(0)
        
        # Extract trade entry and exit points
        current_trade = None
        
        for i in range(1, len(df)):
            date = df.index[i]
            position = df['Position'].iloc[i]
            prev_position = df['Position'].iloc[i-1]
            
            # Check if a new trade started
            if position != 0 and position != prev_position:
                # If there was an open trade, close it
                if current_trade is not None:
                    # Calculate trade profit
                    exit_price = df['Close'].iloc[i]
                    trade_profit = (exit_price / current_trade['entry_price'] - 1) * current_trade['direction']
                    current_trade['exit_date'] = date
                    current_trade['exit_price'] = exit_price
                    current_trade['profit_pct'] = trade_profit * 100  # Convert to percentage
                    
                    # Calculate trade duration
                    if isinstance(date, pd.Timestamp) and isinstance(current_trade['entry_date'], pd.Timestamp):
                        current_trade['duration'] = (date - current_trade['entry_date']).days
                    else:
                        current_trade['duration'] = i - current_trade['entry_idx']
                    
                    trades.append(current_trade)
                
                # Start a new trade
                current_trade = {
                    'entry_date': date,
                    'entry_price': df['Close'].iloc[i],
                    'direction': position,
                    'entry_idx': i
                }
            
            # Check if current trade needs to be closed
            elif position == 0 and prev_position != 0 and current_trade is not None:
                # Calculate trade profit
                exit_price = df['Close'].iloc[i]
                trade_profit = (exit_price / current_trade['entry_price'] - 1) * current_trade['direction']
                current_trade['exit_date'] = date
                current_trade['exit_price'] = exit_price
                current_trade['profit_pct'] = trade_profit * 100  # Convert to percentage
                
                # Calculate trade duration
                if isinstance(date, pd.Timestamp) and isinstance(current_trade['entry_date'], pd.Timestamp):
                    current_trade['duration'] = (date - current_trade['entry_date']).days
                else:
                    current_trade['duration'] = i - current_trade['entry_idx']
                
                trades.append(current_trade)
                current_trade = None
        
        # Close any open trade at the end of the backtest
        if current_trade is not None:
            exit_price = df['Close'].iloc[-1]
            trade_profit = (exit_price / current_trade['entry_price'] - 1) * current_trade['direction']
            current_trade['exit_date'] = df.index[-1]
            current_trade['exit_price'] = exit_price
            current_trade['profit_pct'] = trade_profit * 100  # Convert to percentage
            
            # Calculate trade duration
            if isinstance(df.index[-1], pd.Timestamp) and isinstance(current_trade['entry_date'], pd.Timestamp):
                current_trade['duration'] = (df.index[-1] - current_trade['entry_date']).days
            else:
                current_trade['duration'] = len(df) - 1 - current_trade['entry_idx']
            
            trades.append(current_trade)
        
        return trades
