import pandas as pd
import numpy as np
from datetime import datetime

class CustomBacktester:
    """
    Class for backtesting trading strategies without external dependencies
    """
    
    def __init__(self, market_data):
        """
        Initialize the CustomBacktester class
        
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
        if 'Returns' not in df.columns:
            df['Returns'] = df['Close'].pct_change()
            
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
        if 'Returns' not in df.columns:
            df['Returns'] = df['Close'].pct_change()
            
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
        if 'Returns' not in df.columns:
            df['Returns'] = df['Close'].pct_change()
            
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
    
    def run_macd_strategy(self, position_size=0.1, stop_loss=0.05, take_profit=0.1, max_drawdown=0.25):
        """
        Run MACD strategy backtest
        
        Parameters:
        -----------
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
        if 'Returns' not in df.columns:
            df['Returns'] = df['Close'].pct_change()
            
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
        
        if not signal_columns:
            raise ValueError("No valid indicators provided.")
        
        # Combine signals
        # For combined signals, we sum up all signals and normalize
        df['Combined_Signal'] = 0
        for signal_col in signal_columns:
            df['Combined_Signal'] += df[signal_col]
        
        # Normalize signal to -1, 0, or 1
        df['Signal'] = 0
        df.loc[df['Combined_Signal'] > 0, 'Signal'] = 1
        df.loc[df['Combined_Signal'] < 0, 'Signal'] = -1
        
        # For consensus signals, require all signals to agree
        df['Consensus_Signal'] = 0
        signal_sum = df[signal_columns].sum(axis=1)
        df.loc[signal_sum == len(signal_columns), 'Consensus_Signal'] = 1
        df.loc[signal_sum == -len(signal_columns), 'Consensus_Signal'] = -1
        
        # Use consensus signals if specified
        use_consensus = params.get('use_consensus', False)
        if use_consensus:
            df['Signal'] = df['Consensus_Signal']
        
        # Calculate positions
        df['Position'] = df['Signal'].shift(1)
        df['Position'].fillna(0, inplace=True)
        
        # Calculate returns
        if 'Returns' not in df.columns:
            df['Returns'] = df['Close'].pct_change()
            
        df['Strategy_Returns'] = df['Position'] * df['Returns']
        
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
        # Scale returns by position size
        df['Strategy_Returns'] = df['Strategy_Returns'] * position_size
        
        # Apply stop loss and take profit
        df['Cumulative_Return'] = 0.0
        df['Active_Position'] = False
        df['Entry_Price'] = 0.0
        df['Entry_Index'] = 0
        df['SL_Price'] = 0.0
        df['TP_Price'] = 0.0
        
        current_position = 0
        entry_price = 0
        entry_idx = 0
        sl_price = 0
        tp_price = 0
        
        for i in range(1, len(df)):
            new_position = df['Position'].iloc[i]
            
            # Position changed, record entry
            if new_position != current_position:
                current_position = new_position
                
                if current_position != 0:  # New position
                    entry_price = df['Close'].iloc[i]
                    entry_idx = i
                    
                    # Calculate stop loss and take profit levels
                    if current_position == 1:  # Long
                        sl_price = entry_price * (1 - stop_loss)
                        tp_price = entry_price * (1 + take_profit)
                    else:  # Short
                        sl_price = entry_price * (1 + stop_loss)
                        tp_price = entry_price * (1 - take_profit)
                else:  # No position
                    entry_price = 0
                    sl_price = 0
                    tp_price = 0
            
            # Record current position state
            df['Active_Position'].iloc[i] = (current_position != 0)
            df['Entry_Price'].iloc[i] = entry_price
            df['Entry_Index'].iloc[i] = entry_idx
            df['SL_Price'].iloc[i] = sl_price
            df['TP_Price'].iloc[i] = tp_price
            
            # Check for stop loss or take profit hits
            if current_position != 0:
                # For long positions
                if current_position == 1:
                    # Check stop loss
                    if df['Low'].iloc[i] <= sl_price:
                        stop_loss_return = (sl_price / entry_price - 1) * position_size
                        df['Strategy_Returns'].iloc[i] = stop_loss_return
                        current_position = 0
                        df['Position'].iloc[i] = 0
                    # Check take profit
                    elif df['High'].iloc[i] >= tp_price:
                        take_profit_return = (tp_price / entry_price - 1) * position_size
                        df['Strategy_Returns'].iloc[i] = take_profit_return
                        current_position = 0
                        df['Position'].iloc[i] = 0
                # For short positions
                else:
                    # Check stop loss
                    if df['High'].iloc[i] >= sl_price:
                        stop_loss_return = (entry_price / sl_price - 1) * position_size
                        df['Strategy_Returns'].iloc[i] = stop_loss_return
                        current_position = 0
                        df['Position'].iloc[i] = 0
                    # Check take profit
                    elif df['Low'].iloc[i] <= tp_price:
                        take_profit_return = (entry_price / tp_price - 1) * position_size
                        df['Strategy_Returns'].iloc[i] = take_profit_return
                        current_position = 0
                        df['Position'].iloc[i] = 0
        
        # Calculate equity curve
        df['Equity_Curve'] = (1 + df['Strategy_Returns']).cumprod()
        
        # Calculate drawdown
        df['Peak'] = df['Equity_Curve'].cummax()
        df['Drawdown'] = (df['Equity_Curve'] / df['Peak'] - 1)
        
        # Apply max drawdown rule
        if max_drawdown > 0:
            max_dd_breach = False
            
            for i in range(1, len(df)):
                if df['Drawdown'].iloc[i] <= -max_drawdown:
                    max_dd_breach = True
                    break
            
            if max_dd_breach:
                # Reset positions and returns after the max drawdown breach
                df.loc[df.index[i]:, 'Position'] = 0
                df.loc[df.index[i]:, 'Strategy_Returns'] = 0
                
                # Recalculate equity curve
                df['Equity_Curve'] = (1 + df['Strategy_Returns']).cumprod()
                
                # Recalculate drawdown
                df['Peak'] = df['Equity_Curve'].cummax()
                df['Drawdown'] = (df['Equity_Curve'] / df['Peak'] - 1)
        
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
        # Extract strategy returns
        strategy_returns = df['Strategy_Returns'].dropna()
        
        # Skip if no returns
        if len(strategy_returns) == 0:
            return {
                'total_return': 0,
                'annualized_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'equity_curve': pd.Series([1.0])
            }
        
        # Calculate equity curve
        equity_curve = (1 + strategy_returns).cumprod()
        
        # Calculate total return
        total_return = equity_curve.iloc[-1] - 1 if len(equity_curve) > 0 else 0
        
        # Calculate annualized return (assuming 252 trading days in a year)
        n_days = len(strategy_returns)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate)
        volatility = strategy_returns.std() * np.sqrt(252) if len(strategy_returns) > 0 else 0
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate max drawdown
        peak = equity_curve.cummax()
        drawdown = (equity_curve / peak - 1)
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        # Calculate win rate and profit factor
        winning_trades = strategy_returns[strategy_returns > 0]
        losing_trades = strategy_returns[strategy_returns < 0]
        
        win_rate = len(winning_trades) / len(strategy_returns) if len(strategy_returns) > 0 else 0
        
        total_profits = winning_trades.sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0
        
        profit_factor = total_profits / total_losses if total_losses > 0 else float('inf')
        
        # Create dictionary of metrics
        metrics = {
            'total_return': total_return * 100,  # Convert to percentage
            'annualized_return': annualized_return * 100,  # Convert to percentage
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': abs(max_drawdown) * 100,  # Convert to positive percentage
            'win_rate': win_rate * 100,  # Convert to percentage
            'profit_factor': profit_factor,
            'equity_curve': equity_curve,
            'drawdowns': drawdown
        }
        
        return metrics
    
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
        # Extract position changes
        position_changes = df['Position'].diff().fillna(0)
        
        # Initialize list for storing trades
        trades = []
        
        # Find trade entries and exits
        for i in range(1, len(df)):
            # Entry signal
            if df['Position'].iloc[i-1] == 0 and df['Position'].iloc[i] != 0:
                entry_date = df.index[i]
                entry_price = df['Close'].iloc[i]
                direction = 1 if df['Position'].iloc[i] > 0 else -1
                
                # Look for exit
                for j in range(i+1, len(df)):
                    if df['Position'].iloc[j] == 0 or df['Position'].iloc[j] * direction < 0:
                        exit_date = df.index[j]
                        exit_price = df['Close'].iloc[j]
                        
                        # Calculate profit percentage
                        if direction == 1:  # Long
                            profit_pct = (exit_price / entry_price - 1) * 100
                        else:  # Short
                            profit_pct = (entry_price / exit_price - 1) * 100
                        
                        # Add trade to list
                        trade = {
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'direction': direction,
                            'profit_pct': profit_pct,
                            'duration': (exit_date - entry_date).days
                        }
                        
                        trades.append(trade)
                        break
        
        return trades