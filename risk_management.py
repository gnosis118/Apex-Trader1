import pandas as pd
import numpy as np

class RiskManager:
    """
    Class for managing trading risk
    """
    
    def __init__(self, market_data, position_size=0.1, stop_loss=0.05, take_profit=0.15, max_drawdown=0.25, max_leverage=1.0):
        """
        Initialize the RiskManager class
        
        Parameters:
        -----------
        market_data: pandas.DataFrame
            DataFrame containing market data with technical indicators
        position_size: float
            Default position size as a percentage of portfolio (0.0 to 1.0)
        stop_loss: float
            Default stop loss as a percentage of position value (0.0 to 1.0)
        take_profit: float
            Default take profit as a percentage of position value (0.0 to 1.0)
        max_drawdown: float
            Maximum allowed drawdown percentage (0.0 to 1.0)
        max_leverage: float
            Maximum allowed leverage (>= 1.0)
        """
        self.data = market_data.copy()
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_drawdown = max_drawdown
        self.max_leverage = max_leverage
        
        # Calculate volatility measures for dynamic sizing
        self._calculate_volatility_metrics()
    
    def _calculate_volatility_metrics(self):
        """
        Calculate volatility metrics used for risk management
        """
        # Calculate return series
        if 'Returns' not in self.data.columns:
            self.data['Returns'] = self.data['Close'].pct_change()
            
        # Calculate rolling volatility (20-day)
        self.data['Volatility_20d'] = self.data['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Calculate Average True Range if not already present
        if 'ATR_14' not in self.data.columns:
            try:
                import talib as ta
                self.data['ATR_14'] = ta.ATR(
                    self.data['High'].values,
                    self.data['Low'].values,
                    self.data['Close'].values,
                    timeperiod=14
                )
            except:
                # Calculate ATR manually if talib is not available
                tr1 = self.data['High'] - self.data['Low']
                tr2 = abs(self.data['High'] - self.data['Close'].shift())
                tr3 = abs(self.data['Low'] - self.data['Close'].shift())
                tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
                self.data['ATR_14'] = tr.rolling(14).mean()
    
    def calculate_position_size(self, volatility_scaling=True, atr_multiplier=2.0):
        """
        Calculate appropriate position size based on market conditions
        
        Parameters:
        -----------
        volatility_scaling: bool
            Whether to scale position size based on volatility
        atr_multiplier: float
            Multiplier for ATR-based position sizing
            
        Returns:
        --------
        float
            The calculated position size (0.0 to 1.0)
        """
        current_volatility = self.data['Volatility_20d'].iloc[-1]
        
        if np.isnan(current_volatility):
            return self.position_size
        
        if volatility_scaling:
            # Calculate volatility-adjusted position size
            # Higher volatility = smaller position size
            historical_vol_median = self.data['Volatility_20d'].median()
            vol_ratio = historical_vol_median / current_volatility if current_volatility > 0 else 1.0
            
            # Scale position size based on volatility
            scaled_position = self.position_size * min(vol_ratio, 2.0)  # Cap scaling at 2x
            
            # Ensure position size is within bounds
            position_size = min(scaled_position, self.position_size * 2.0)
            position_size = max(position_size, self.position_size * 0.5)
            
            return position_size
        else:
            return self.position_size
    
    def calculate_stop_loss(self, price, direction, atr_based=True, atr_multiplier=2.0):
        """
        Calculate stop loss price based on current market conditions
        
        Parameters:
        -----------
        price: float
            Current price
        direction: int
            Trade direction (1 for long, -1 for short)
        atr_based: bool
            Whether to use ATR-based stop loss
        atr_multiplier: float
            Multiplier for ATR-based stop loss
            
        Returns:
        --------
        float
            The calculated stop loss price
        """
        if atr_based and 'ATR_14' in self.data.columns:
            current_atr = self.data['ATR_14'].iloc[-1]
            
            if not np.isnan(current_atr):
                # ATR-based stop loss
                stop_distance = current_atr * atr_multiplier
                
                # Calculate stop price
                if direction > 0:  # Long position
                    stop_price = price * (1 - stop_distance / price)
                else:  # Short position
                    stop_price = price * (1 + stop_distance / price)
                
                return stop_price
        
        # Default percentage-based stop loss
        if direction > 0:  # Long position
            return price * (1 - self.stop_loss)
        else:  # Short position
            return price * (1 + self.stop_loss)
    
    def calculate_take_profit(self, price, direction, risk_reward_ratio=2.0):
        """
        Calculate take profit price based on risk-reward ratio
        
        Parameters:
        -----------
        price: float
            Current price
        direction: int
            Trade direction (1 for long, -1 for short)
        risk_reward_ratio: float
            Target risk-reward ratio (take_profit distance / stop_loss distance)
            
        Returns:
        --------
        float
            The calculated take profit price
        """
        stop_price = self.calculate_stop_loss(price, direction)
        stop_distance = abs(price - stop_price)
        take_profit_distance = stop_distance * risk_reward_ratio
        
        if direction > 0:  # Long position
            return price + take_profit_distance
        else:  # Short position
            return price - take_profit_distance
    
    def calculate_value_at_risk(self, portfolio_value, confidence_level=0.95, time_horizon=1):
        """
        Calculate Value at Risk (VaR) for the portfolio
        
        Parameters:
        -----------
        portfolio_value: float
            Current portfolio value
        confidence_level: float
            Confidence level for VaR calculation (0.0 to 1.0)
        time_horizon: int
            Time horizon in days
            
        Returns:
        --------
        float
            The calculated VaR value
        """
        returns = self.data['Returns'].dropna()
        
        if len(returns) < 30:
            # Not enough data for reliable VaR calculation
            return portfolio_value * self.position_size * self.stop_loss
        
        # Calculate VaR using historical method
        var_percentile = 1.0 - confidence_level
        var_daily = np.percentile(returns, var_percentile * 100)
        
        # Scale to time horizon
        var_horizon = var_daily * np.sqrt(time_horizon)
        
        return portfolio_value * abs(var_horizon)
    
    def calculate_expected_shortfall(self, portfolio_value, confidence_level=0.95):
        """
        Calculate Expected Shortfall (ES) / Conditional VaR
        
        Parameters:
        -----------
        portfolio_value: float
            Current portfolio value
        confidence_level: float
            Confidence level for ES calculation (0.0 to 1.0)
            
        Returns:
        --------
        float
            The calculated ES value
        """
        returns = self.data['Returns'].dropna()
        
        if len(returns) < 30:
            # Not enough data for reliable ES calculation
            return portfolio_value * self.position_size * self.stop_loss * 1.5
        
        # Calculate VaR percentile
        var_percentile = 1.0 - confidence_level
        var_cutoff = np.percentile(returns, var_percentile * 100)
        
        # Calculate Expected Shortfall
        es = returns[returns <= var_cutoff].mean()
        
        return portfolio_value * abs(es)
    
    def check_max_drawdown(self, equity_curve):
        """
        Check if maximum drawdown threshold has been exceeded
        
        Parameters:
        -----------
        equity_curve: pandas.Series
            Series containing equity curve values
            
        Returns:
        --------
        tuple
            (bool: exceeds max drawdown, float: current drawdown percentage)
        """
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve / peak - 1)
        current_drawdown = abs(drawdown.min()) if drawdown.min() < 0 else 0
        
        return current_drawdown > self.max_drawdown, current_drawdown
    
    def kelly_criterion(self, win_rate, win_loss_ratio):
        """
        Calculate optimal position size using Kelly Criterion
        
        Parameters:
        -----------
        win_rate: float
            Probability of winning (0.0 to 1.0)
        win_loss_ratio: float
            Ratio of average win to average loss
            
        Returns:
        --------
        float
            Optimal position size (0.0 to 1.0)
        """
        if win_loss_ratio <= 0 or win_rate <= 0:
            return 0.0
        
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Usually only a fraction of Kelly is used (half or quarter Kelly)
        half_kelly = kelly * 0.5
        
        # Ensure position size is within bounds
        position_size = min(max(half_kelly, 0.01), self.position_size)
        return position_size
    
    def adjust_for_correlation(self, position_size, correlation):
        """
        Adjust position size based on correlation with existing positions
        
        Parameters:
        -----------
        position_size: float
            Calculated position size
        correlation: float
            Correlation with existing positions (-1.0 to 1.0)
            
        Returns:
        --------
        float
            Adjusted position size
        """
        # Increase position size for negatively correlated assets
        # Decrease position size for positively correlated assets
        if correlation < 0:
            adjusted_size = position_size * (1 + abs(correlation) * 0.3)
        else:
            adjusted_size = position_size * (1 - correlation * 0.3)
        
        return min(adjusted_size, self.position_size * 1.5)
