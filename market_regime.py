import numpy as np
import pandas as pd

class MarketRegimeDetector:
    """
    Detects market regimes such as trending, ranging, volatile, etc.
    Uses multiple factors to determine the current market state.
    """
    
    def __init__(self):
        """Initialize the regime detector"""
        self.regime_history = []
    
    def detect_regime(self, data, lookback_period=20):
        """
        Detect the current market regime based on price action
        
        Parameters:
        -----------
        data: pandas.DataFrame
            Market data with OHLCV and indicators
        lookback_period: int
            Period to analyze for regime detection
            
        Returns:
        --------
        dict
            Regime classification with confidence levels
        """
        # Ensure we have enough data
        if len(data) < lookback_period:
            return {"regime": "UNKNOWN", "confidence": 0.0, "sub_regimes": {}}
        
        # Get recent data for analysis
        recent_data = data.iloc[-lookback_period:]
        
        # Calculate regime indicators
        regime_indicators = {}
        
        # 1. Trend strength
        if 'Close' in recent_data.columns:
            close_prices = recent_data['Close']
            price_change = (close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]
            std_dev = close_prices.std() / close_prices.mean()
            
            # Simple trend detection
            ma_short = close_prices.rolling(window=5).mean()
            ma_long = close_prices.rolling(window=lookback_period).mean()
            
            # Trend strength measured by difference between short and long MA
            trend_strength = (ma_short.iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1]
            
            # Normalize between -1 and 1
            trend_strength = np.clip(trend_strength * 10, -1, 1)
            regime_indicators['trend_strength'] = trend_strength
            
            # Direction of trend
            trend_direction = np.sign(trend_strength)
            regime_indicators['trend_direction'] = trend_direction
        else:
            regime_indicators['trend_strength'] = 0
            regime_indicators['trend_direction'] = 0
        
        # 2. Volatility regime
        if 'High' in recent_data.columns and 'Low' in recent_data.columns:
            # Calculate ATR-like measure
            high_low_range = recent_data['High'] - recent_data['Low']
            atr = high_low_range.mean()
            avg_price = recent_data['Close'].mean()
            
            # Normalize volatility measure
            volatility = atr / avg_price
            
            # Compare to recent history to determine if volatility is high/low
            # This is a simplification - in a full implementation would compare to longer history
            volatility_threshold = 0.02  # 2% range as moderate volatility threshold
            
            if volatility > volatility_threshold * 1.5:
                volatility_regime = "HIGH_VOLATILITY"
                volatility_score = 1.0
            elif volatility < volatility_threshold * 0.5:
                volatility_regime = "LOW_VOLATILITY"
                volatility_score = -1.0
            else:
                volatility_regime = "NORMAL_VOLATILITY"
                volatility_score = 0.0
                
            regime_indicators['volatility'] = volatility_score
        else:
            regime_indicators['volatility'] = 0
        
        # 3. Range vs Trend Detection
        if 'Close' in recent_data.columns:
            # Calculate price movements
            returns = recent_data['Close'].pct_change().dropna()
            
            # Check for directional consistency - measure of trending behavior
            pos_returns = (returns > 0).sum()
            neg_returns = (returns < 0).sum()
            
            # Directional consistency ratio
            total_moves = pos_returns + neg_returns
            if total_moves > 0:
                directional_consistency = max(pos_returns, neg_returns) / total_moves
            else:
                directional_consistency = 0.5
            
            # Normalize to -1 to 1 scale, with 0 being random, 1 being consistent trend
            ranging_vs_trending = (directional_consistency - 0.5) * 2
            regime_indicators['ranging_vs_trending'] = ranging_vs_trending
        else:
            regime_indicators['ranging_vs_trending'] = 0
        
        # Now determine overall regime
        if regime_indicators['trend_strength'] > 0.6 and regime_indicators['ranging_vs_trending'] > 0.3:
            main_regime = "TRENDING_UP"
            confidence = regime_indicators['trend_strength'] * regime_indicators['ranging_vs_trending']
        elif regime_indicators['trend_strength'] < -0.6 and regime_indicators['ranging_vs_trending'] > 0.3:
            main_regime = "TRENDING_DOWN"
            confidence = -regime_indicators['trend_strength'] * regime_indicators['ranging_vs_trending']
        elif abs(regime_indicators['trend_strength']) < 0.3 and regime_indicators['ranging_vs_trending'] < 0:
            main_regime = "RANGING"
            confidence = -regime_indicators['ranging_vs_trending'] * (1 - abs(regime_indicators['trend_strength']))
        elif regime_indicators['volatility'] > 0.7:
            main_regime = "VOLATILE"
            confidence = regime_indicators['volatility']
        else:
            main_regime = "MIXED"
            confidence = 0.5
        
        # Construct result
        result = {
            "regime": main_regime,
            "confidence": float(confidence),
            "indicators": regime_indicators
        }
        
        self.regime_history.append(result)
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]
            
        return result
    
    def get_optimal_strategy(self, regime):
        """
        Return the optimal strategy parameters for a given market regime
        
        Parameters:
        -----------
        regime: dict
            The detected market regime
            
        Returns:
        --------
        dict
            Strategy parameters optimized for the regime
        """
        regime_type = regime["regime"]
        confidence = regime["confidence"]
        
        # Default parameters
        default_params = {
            "use_ma": True,
            "use_rsi": True,
            "fast_ma": 20,
            "slow_ma": 50,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "position_size": 0.1,
            "stop_loss": 0.05,
            "take_profit": 0.1,
            "max_drawdown": 0.25
        }
        
        # Adjust parameters based on regime
        if regime_type == "TRENDING_UP":
            # For uptrends, trend-following with looser exits
            return {
                "use_ma": True,
                "use_rsi": False,  # Reduce mean-reversion in trend
                "fast_ma": 10,
                "slow_ma": 30,
                "position_size": min(0.15, default_params["position_size"] * (1 + confidence)),
                "stop_loss": default_params["stop_loss"] * (1 - confidence * 0.3),  # Wider stops in strong trend
                "take_profit": default_params["take_profit"] * (1 + confidence * 0.5)  # Larger targets in trend
            }
        
        elif regime_type == "TRENDING_DOWN":
            # For downtrends, trend-following with tighter stops
            return {
                "use_ma": True,
                "use_rsi": False,
                "fast_ma": 10,
                "slow_ma": 30,
                "position_size": min(0.15, default_params["position_size"] * (1 + confidence * 0.7)),
                "stop_loss": default_params["stop_loss"] * (1 - confidence * 0.1),  # Still need protection
                "take_profit": default_params["take_profit"] * (1 + confidence * 0.3)
            }
        
        elif regime_type == "RANGING":
            # For ranges, focus on mean reversion with RSI
            return {
                "use_ma": False,
                "use_rsi": True,
                "rsi_period": 10,  # More responsive
                "rsi_overbought": 70 - (confidence * 5),  # More aggressive entries
                "rsi_oversold": 30 + (confidence * 5),
                "position_size": max(0.05, default_params["position_size"] * (1 - confidence * 0.3)),  # Smaller size
                "stop_loss": default_params["stop_loss"] * (1 + confidence * 0.2),  # Need wider stops in ranges
                "take_profit": default_params["take_profit"] * (1 - confidence * 0.4)  # Smaller targets in ranges
            }
        
        elif regime_type == "VOLATILE":
            # For volatile markets, reduce risk
            return {
                "use_ma": True,
                "use_rsi": True,
                "fast_ma": 15,
                "slow_ma": 40,
                "rsi_period": 14,
                "rsi_overbought": 75,  # More conservative
                "rsi_oversold": 25,
                "position_size": max(0.03, default_params["position_size"] * (1 - confidence * 0.7)),  # Much smaller
                "stop_loss": default_params["stop_loss"] * (1 + confidence * 0.5),  # Wider stops for volatility
                "take_profit": default_params["take_profit"] * (1 + confidence * 0.2)
            }
        
        else:  # MIXED or UNKNOWN
            # Use default parameters with slightly reduced risk
            default_params["position_size"] *= 0.8
            return default_params
            
    def get_regime_history(self):
        """
        Return the history of detected regimes
        
        Returns:
        --------
        list
            List of regime dictionaries
        """
        return self.regime_history


def analyze_market_structure(data, period=20):
    """
    Analyze market structure to identify support/resistance and key levels
    
    Parameters:
    -----------
    data: pandas.DataFrame
        Market data with OHLCV
    period: int
        Lookback period for analysis
        
    Returns:
    --------
    dict
        Market structure information
    """
    # Simple implementation - in real system would be more sophisticated
    if len(data) < period:
        return {"supports": [], "resistances": [], "key_levels": []}
    
    recent_data = data.iloc[-period:]
    
    # Find swing highs and lows
    highs = recent_data['High'].values
    lows = recent_data['Low'].values
    
    swing_highs = []
    swing_lows = []
    
    # Very simple swing detection - in production would use more advanced methods
    for i in range(2, len(highs) - 2):
        # Swing high
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append((i, highs[i]))
        
        # Swing low
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append((i, lows[i]))
    
    # Group nearby levels
    def group_levels(levels, threshold=0.005):
        if not levels:
            return []
            
        # Sort by price
        sorted_levels = sorted(levels, key=lambda x: x[1])
        
        # Group nearby levels
        grouped = []
        current_group = [sorted_levels[0]]
        
        for i in range(1, len(sorted_levels)):
            current_level = sorted_levels[i]
            prev_level = sorted_levels[i-1]
            
            # If close to previous level, add to current group
            if abs(current_level[1] - prev_level[1]) / prev_level[1] < threshold:
                current_group.append(current_level)
            else:
                # Calculate average price for the group
                avg_price = sum(l[1] for l in current_group) / len(current_group)
                grouped.append(avg_price)
                current_group = [current_level]
        
        # Add the last group
        if current_group:
            avg_price = sum(l[1] for l in current_group) / len(current_group)
            grouped.append(avg_price)
            
        return grouped
    
    # Group supports and resistances
    supports = group_levels(swing_lows)
    resistances = group_levels(swing_highs)
    
    # Combine into key levels (both support and resistance)
    all_levels = supports + resistances
    key_levels = group_levels([(0, level) for level in all_levels], threshold=0.01)
    
    return {
        "supports": supports,
        "resistances": resistances,
        "key_levels": key_levels
    }