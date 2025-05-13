import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from custom_data_processor import CustomDataProcessor
from custom_ta import CustomTA
from custom_backtester import CustomBacktester
from custom_risk_management import CustomRiskManager
from custom_visualization import (
    create_candlestick_chart, 
    create_technical_indicator_chart, 
    create_performance_chart, 
    create_returns_heatmap,
    create_trade_distribution_chart
)

# Set page configuration
st.set_page_config(
    page_title="Futures Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'selected_timeframe' not in st.session_state:
    st.session_state.selected_timeframe = '1d'
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 'Dashboard'

# Header
st.title("Self-Learning Futures Trading Bot")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Navigation
    st.session_state.current_tab = st.radio(
        "Navigation", 
        ['Dashboard', 'Data Analysis', 'Backtesting', 'Performance']
    )
    
    # Symbol selection
    symbol = st.selectbox(
        "Select Futures Contract",
        ["ES=F", "NQ=F", "YM=F", "RTY=F", "CL=F", "GC=F", "SI=F", "ZC=F", "ZS=F", "6E=F", "6J=F", "BTC-USD"],
        index=0
    )
    
    # Timeframe selection
    timeframe = st.selectbox(
        "Select Timeframe",
        ["1d", "4h", "1h", "30m", "15m", "5m", "1m"],
        index=0
    )
    st.session_state.selected_timeframe = timeframe
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            datetime.now() - timedelta(days=365)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            datetime.now()
        )
    
    # Load data button
    if st.button("Load Data"):
        with st.spinner("Loading market data..."):
            try:
                # Initialize data processor
                st.session_state.data_processor = CustomDataProcessor(symbol, timeframe, start_date, end_date)
                st.session_state.market_data = st.session_state.data_processor.get_data()
                
                # Initialize technical analysis
                st.session_state.ta = CustomTA(st.session_state.market_data)
                st.session_state.market_data_with_indicators = st.session_state.ta.add_all_indicators()
                
                st.session_state.data_loaded = True
                st.success(f"Successfully loaded data for {symbol}")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    if st.session_state.data_loaded:
        st.success("✅ Data loaded")
        
        # Strategy parameters section
        st.subheader("Strategy Parameters")
        
        # Technical indicator parameters
        st.session_state.ma_fast = st.slider("Fast MA Period", 5, 50, 20)
        st.session_state.ma_slow = st.slider("Slow MA Period", 20, 200, 50)
        st.session_state.rsi_period = st.slider("RSI Period", 7, 30, 14)
        st.session_state.rsi_overbought = st.slider("RSI Overbought", 70, 90, 70)
        st.session_state.rsi_oversold = st.slider("RSI Oversold", 10, 30, 30)
        st.session_state.bb_period = st.slider("Bollinger Bands Period", 10, 50, 20)
        st.session_state.bb_std = st.slider("Bollinger Bands Std Dev", 1.0, 3.0, 2.0, 0.1)
        
        # Risk management parameters
        st.subheader("Risk Management")
        st.session_state.position_size = st.slider("Position Size (%)", 1, 100, 10) / 100
        st.session_state.stop_loss = st.slider("Stop Loss (%)", 1, 20, 5) / 100
        st.session_state.take_profit = st.slider("Take Profit (%)", 1, 50, 15) / 100
        st.session_state.max_drawdown = st.slider("Max Drawdown (%)", 5, 50, 25) / 100

# Main content area based on selected tab
if not st.session_state.data_loaded:
    # Show sample dashboard with placeholders
    st.header("Welcome to the Futures Trading Bot")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://pixabay.com/get/gc73bb9e10face2db12906e9d4a3d077611cf0f39fe97ae7f212a436d875bd97d7790de2b7e0bfd6785e574afa98cae7978112475e6185e817d4d29b3c4f23689_1280.jpg", 
                 caption="Financial Trading Dashboard")
    with col2:
        st.image("https://pixabay.com/get/g34c941277dd030616d5c1ccacaccf7ccabeab9e9a0e147b7265c0e8c527bd2546313d0340ae9a672cc3a6a4ef031636f4a9b975f3a400a02b1811f3e1e92a0cf_1280.jpg", 
                 caption="Market Data Analysis")
    
    st.subheader("Getting Started")
    st.write("""
    1. Select a futures contract from the sidebar
    2. Choose a timeframe for analysis
    3. Set the date range for historical data
    4. Click 'Load Data' to begin
    
    This trading bot provides:
    - Historical market data visualization
    - Technical analysis with popular indicators
    - Strategy backtesting and optimization
    - Performance analytics and risk management
    """)
    
    st.subheader("Key Features")
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    with feature_col1:
        st.markdown("### Technical Analysis")
        st.write("MACD, RSI, Bollinger Bands, Moving Averages")
    with feature_col2:
        st.markdown("### Backtesting")
        st.write("Test strategies against historical data")
    with feature_col3:
        st.markdown("### Risk Management")
        st.write("Position sizing, stop-loss, and drawdown control")
        
else:
    # Dashboard with real data
    if st.session_state.current_tab == 'Dashboard':
        st.header("Trading Dashboard")
        
        # Overview metrics
        metric1, metric2, metric3, metric4 = st.columns(4)
        
        with metric1:
            last_price = st.session_state.market_data.iloc[-1]['Close']
            previous_price = st.session_state.market_data.iloc[-2]['Close']
            price_change = ((last_price - previous_price) / previous_price) * 100
            st.metric("Last Price", f"{last_price:.2f}", f"{price_change:.2f}%")
            
        with metric2:
            volume = st.session_state.market_data.iloc[-1]['Volume']
            avg_volume = st.session_state.market_data['Volume'].mean()
            volume_change = ((volume - avg_volume) / avg_volume) * 100
            st.metric("Volume", f"{volume:.0f}", f"{volume_change:.2f}%")
            
        with metric3:
            if 'RSI_14' in st.session_state.market_data_with_indicators.columns:
                rsi = st.session_state.market_data_with_indicators.iloc[-1]['RSI_14']
                st.metric("RSI", f"{rsi:.2f}", "")
            else:
                st.metric("RSI", "N/A", "")
                
        with metric4:
            if 'Volatility' in st.session_state.market_data_with_indicators.columns:
                volatility = st.session_state.market_data_with_indicators.iloc[-1]['Volatility']
                st.metric("Volatility", f"{volatility:.2f}%", "")
            else:
                # Calculate a simple volatility metric if not available
                returns = st.session_state.market_data['Returns'].dropna()
                volatility = returns.std() * 100 * np.sqrt(252)  # Annualized volatility
                st.metric("Volatility", f"{volatility:.2f}%", "")
        
        # Main chart
        st.subheader("Price Chart with Technical Indicators")
        
        # Create and display the candlestick chart
        fig = create_candlestick_chart(
            st.session_state.market_data_with_indicators,
            ma_fast=st.session_state.ma_fast,
            ma_slow=st.session_state.ma_slow,
            bb_period=st.session_state.bb_period,
            bb_std=st.session_state.bb_std
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical indicators
        st.subheader("Technical Indicators")
        
        # Create tabs for different indicators
        tab1, tab2, tab3 = st.tabs(["Momentum", "Volatility", "Trend"])
        
        with tab1:
            # RSI and MACD
            rsi_fig = create_technical_indicator_chart(
                st.session_state.market_data_with_indicators,
                indicator_type='momentum'
            )
            st.plotly_chart(rsi_fig, use_container_width=True)
            
        with tab2:
            # Bollinger Bands Width and ATR
            vol_fig = create_technical_indicator_chart(
                st.session_state.market_data_with_indicators,
                indicator_type='volatility'
            )
            st.plotly_chart(vol_fig, use_container_width=True)
            
        with tab3:
            # Moving Averages
            trend_fig = create_technical_indicator_chart(
                st.session_state.market_data_with_indicators,
                indicator_type='trend'
            )
            st.plotly_chart(trend_fig, use_container_width=True)
            
    # Data Analysis Tab    
    elif st.session_state.current_tab == 'Data Analysis':
        st.header("Data Analysis")
        
        # Basic statistics
        st.subheader("Market Data Statistics")
        
        # Create a summary dataframe
        summary_df = pd.DataFrame({
            'Open': [st.session_state.market_data['Open'].min(), st.session_state.market_data['Open'].max(), 
                    st.session_state.market_data['Open'].mean(), st.session_state.market_data['Open'].std()],
            'High': [st.session_state.market_data['High'].min(), st.session_state.market_data['High'].max(), 
                    st.session_state.market_data['High'].mean(), st.session_state.market_data['High'].std()],
            'Low': [st.session_state.market_data['Low'].min(), st.session_state.market_data['Low'].max(), 
                   st.session_state.market_data['Low'].mean(), st.session_state.market_data['Low'].std()],
            'Close': [st.session_state.market_data['Close'].min(), st.session_state.market_data['Close'].max(), 
                     st.session_state.market_data['Close'].mean(), st.session_state.market_data['Close'].std()],
            'Volume': [st.session_state.market_data['Volume'].min(), st.session_state.market_data['Volume'].max(), 
                      st.session_state.market_data['Volume'].mean(), st.session_state.market_data['Volume'].std()]
        }, index=['Min', 'Max', 'Mean', 'Std Dev'])
        
        st.dataframe(summary_df, use_container_width=True)
        
        # Returns analysis
        st.subheader("Returns Analysis")
        
        returns = st.session_state.market_data['Returns'].dropna()
        
        # Plot returns distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=returns, nbinsx=50, name='Returns', marker_color='blue'))
        fig.update_layout(
            title='Returns Distribution',
            xaxis_title='Daily Returns',
            yaxis_title='Frequency',
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics about returns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Return", f"{returns.mean()*100:.4f}%")
        with col2:
            st.metric("Return Std Dev", f"{returns.std()*100:.4f}%")
        with col3:
            st.metric("Skewness", f"{returns.skew():.4f}")
        with col4:
            st.metric("Kurtosis", f"{returns.kurtosis():.4f}")
            
        # Correlation matrix of technical indicators
        st.subheader("Technical Indicator Correlation")
        
        # Select only numeric columns for correlation
        numeric_cols = st.session_state.market_data_with_indicators.select_dtypes(include=['float64', 'int64']).columns
        correlation = st.session_state.market_data_with_indicators[numeric_cols].corr()
        
        # Plot correlation heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation.values,
            x=correlation.columns,
            y=correlation.columns,
            colorscale='Viridis',
            zmin=-1, zmax=1
        ))
        fig.update_layout(
            title='Correlation Matrix of Technical Indicators',
            template='plotly_dark',
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
    # Backtesting Tab
    elif st.session_state.current_tab == 'Backtesting':
        st.header("Strategy Backtesting")
        
        # Strategy selection
        strategy_type = st.selectbox(
            "Select Strategy Type",
            ["Moving Average Crossover", "RSI Mean Reversion", "Bollinger Bands", "MACD", "Combined Strategy"]
        )
        
        # Strategy parameters based on selection
        if strategy_type == "Moving Average Crossover":
            st.subheader("Moving Average Crossover Parameters")
            fast_ma = st.slider("Fast MA Period", 5, 50, st.session_state.ma_fast, key="bt_fast_ma")
            slow_ma = st.slider("Slow MA Period", 20, 200, st.session_state.ma_slow, key="bt_slow_ma")
            
        elif strategy_type == "RSI Mean Reversion":
            st.subheader("RSI Mean Reversion Parameters")
            rsi_period = st.slider("RSI Period", 7, 30, st.session_state.rsi_period, key="bt_rsi_period")
            rsi_overbought = st.slider("Overbought Level", 60, 90, st.session_state.rsi_overbought, key="bt_rsi_overbought")
            rsi_oversold = st.slider("Oversold Level", 10, 40, st.session_state.rsi_oversold, key="bt_rsi_oversold")
            
        elif strategy_type == "Bollinger Bands":
            st.subheader("Bollinger Bands Parameters")
            bb_period = st.slider("BB Period", 10, 50, st.session_state.bb_period, key="bt_bb_period")
            bb_std = st.slider("BB Standard Deviation", 1.0, 3.0, st.session_state.bb_std, key="bt_bb_std")
            
        elif strategy_type == "MACD":
            st.subheader("MACD Parameters")
            macd_fast = st.slider("MACD Fast Period", 8, 20, 12, key="bt_macd_fast")
            macd_slow = st.slider("MACD Slow Period", 20, 40, 26, key="bt_macd_slow")
            macd_signal = st.slider("MACD Signal Period", 5, 15, 9, key="bt_macd_signal")
            
        elif strategy_type == "Combined Strategy":
            st.subheader("Combined Strategy Parameters")
            # Let user select which indicators to include
            use_ma = st.checkbox("Use Moving Averages", True)
            use_rsi = st.checkbox("Use RSI", True)
            use_bb = st.checkbox("Use Bollinger Bands", True)
            use_macd = st.checkbox("Use MACD", True)
            
            # Only show parameters for selected indicators
            indicators = []
            params = {}
            
            if use_ma:
                indicators.append('ma_crossover')
                params['fast_ma'] = st.slider("Fast MA Period", 5, 50, st.session_state.ma_fast, key="comb_fast_ma")
                params['slow_ma'] = st.slider("Slow MA Period", 20, 200, st.session_state.ma_slow, key="comb_slow_ma")
            
            if use_rsi:
                indicators.append('rsi')
                params['rsi_period'] = st.slider("RSI Period", 7, 30, st.session_state.rsi_period, key="comb_rsi_period")
                params['rsi_overbought'] = st.slider("RSI Overbought", 60, 90, st.session_state.rsi_overbought, key="comb_rsi_overbought")
                params['rsi_oversold'] = st.slider("RSI Oversold", 10, 40, st.session_state.rsi_oversold, key="comb_rsi_oversold")
            
            if use_bb:
                indicators.append('bollinger_bands')
                params['bb_period'] = st.slider("BB Period", 10, 50, st.session_state.bb_period, key="comb_bb_period")
                params['bb_std'] = st.slider("BB Standard Deviation", 1.0, 3.0, st.session_state.bb_std, key="comb_bb_std")
            
            if use_macd:
                indicators.append('macd')
                params['macd_fast'] = st.slider("MACD Fast", 8, 20, 12, key="comb_macd_fast")
                params['macd_slow'] = st.slider("MACD Slow", 20, 40, 26, key="comb_macd_slow")
                params['macd_signal'] = st.slider("MACD Signal", 5, 15, 9, key="comb_macd_signal")
            
            params['use_consensus'] = st.checkbox("Require Consensus", False)
        
        # Run backtest button
        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                try:
                    # Initialize backtester
                    backtester = CustomBacktester(st.session_state.market_data_with_indicators)
                    
                    # Run selected strategy
                    if strategy_type == "Moving Average Crossover":
                        results = backtester.run_ma_crossover_strategy(
                            fast_period=fast_ma,
                            slow_period=slow_ma,
                            position_size=st.session_state.position_size,
                            stop_loss=st.session_state.stop_loss,
                            take_profit=st.session_state.take_profit,
                            max_drawdown=st.session_state.max_drawdown
                        )
                    elif strategy_type == "RSI Mean Reversion":
                        results = backtester.run_rsi_strategy(
                            rsi_period=rsi_period,
                            overbought=rsi_overbought,
                            oversold=rsi_oversold,
                            position_size=st.session_state.position_size,
                            stop_loss=st.session_state.stop_loss,
                            take_profit=st.session_state.take_profit,
                            max_drawdown=st.session_state.max_drawdown
                        )
                    elif strategy_type == "Bollinger Bands":
                        results = backtester.run_bollinger_bands_strategy(
                            bb_period=bb_period,
                            bb_std=bb_std,
                            position_size=st.session_state.position_size,
                            stop_loss=st.session_state.stop_loss,
                            take_profit=st.session_state.take_profit,
                            max_drawdown=st.session_state.max_drawdown
                        )
                    elif strategy_type == "MACD":
                        results = backtester.run_macd_strategy(
                            position_size=st.session_state.position_size,
                            stop_loss=st.session_state.stop_loss,
                            take_profit=st.session_state.take_profit,
                            max_drawdown=st.session_state.max_drawdown
                        )
                    elif strategy_type == "Combined Strategy":
                        results = backtester.run_combined_strategy(
                            indicators=indicators,
                            params=params,
                            position_size=st.session_state.position_size,
                            stop_loss=st.session_state.stop_loss,
                            take_profit=st.session_state.take_profit,
                            max_drawdown=st.session_state.max_drawdown
                        )
                    
                    # Store results in session state
                    st.session_state.backtest_results = results
                    st.success("Backtest completed successfully!")
                except Exception as e:
                    st.error(f"Error running backtest: {str(e)}")
        
        # Display backtest results if available
        if st.session_state.backtest_results:
            results = st.session_state.backtest_results
            
            # Performance metrics
            st.subheader("Performance Metrics")
            
            # Row 1: Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{results['total_return']:.2f}%")
            with col2:
                st.metric("Annualized Return", f"{results['annualized_return']:.2f}%")
            with col3:
                st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
            with col4:
                st.metric("Max Drawdown", f"{results['max_drawdown']:.2f}%")
            
            # Row 2: Trade metrics
            col1, col2, col3, col4 = st.columns(4)
            
            # Calculate win rate and profit factor if trades exist
            trades = results.get('trades', [])
            win_rate = 0
            profit_factor = 0
            
            if trades:
                win_count = sum(1 for trade in trades if trade['profit_pct'] > 0)
                win_rate = (win_count / len(trades)) * 100
                
                total_profit = sum(trade['profit_pct'] for trade in trades if trade['profit_pct'] > 0)
                total_loss = abs(sum(trade['profit_pct'] for trade in trades if trade['profit_pct'] <= 0))
                profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            with col1:
                st.metric("Number of Trades", len(trades))
            with col2:
                st.metric("Win Rate", f"{win_rate:.2f}%")
            with col3:
                st.metric("Profit Factor", f"{profit_factor:.2f}")
            with col4:
                avg_hold_time = sum(trade.get('duration', 0) for trade in trades) / len(trades) if trades else 0
                st.metric("Avg Hold Time", f"{avg_hold_time:.1f} days")
            
            # Performance chart
            st.subheader("Equity Curve")
            
            equity_fig = create_performance_chart(
                results['equity_curve'],
                results['drawdowns'],
                chart_type='equity_drawdown'
            )
            st.plotly_chart(equity_fig, use_container_width=True)
            
            # Trade distribution
            st.subheader("Trade Distribution")
            
            trade_fig = create_trade_distribution_chart(trades)
            st.plotly_chart(trade_fig, use_container_width=True)
            
            # Price chart with buy/sell signals
            st.subheader("Price Chart with Signals")
            
            signal_fig = create_candlestick_chart(
                st.session_state.market_data_with_indicators,
                ma_fast=st.session_state.ma_fast,
                ma_slow=st.session_state.ma_slow,
                bb_period=st.session_state.bb_period,
                bb_std=st.session_state.bb_std,
                buy_signals=results.get('buy_signals', []),
                sell_signals=results.get('sell_signals', [])
            )
            st.plotly_chart(signal_fig, use_container_width=True)
            
            # Trade list
            if trades:
                st.subheader("List of Trades")
                
                # Convert trades list to DataFrame
                trades_df = pd.DataFrame(trades)
                
                # Format the DataFrame
                if 'entry_date' in trades_df.columns:
                    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
                if 'exit_date' in trades_df.columns:
                    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
                if 'direction' in trades_df.columns:
                    trades_df['direction'] = trades_df['direction'].map({1: 'Long', -1: 'Short'})
                if 'profit_pct' in trades_df.columns:
                    trades_df['profit_pct'] = trades_df['profit_pct'].round(2).astype(str) + '%'
                
                # Display the DataFrame
                st.dataframe(trades_df, use_container_width=True)
    
    # Performance Tab
    elif st.session_state.current_tab == 'Performance':
        st.header("Performance Analytics")
        
        if not st.session_state.backtest_results:
            st.info("Run a backtest first to see performance analytics.")
        else:
            results = st.session_state.backtest_results
            
            # Monthly returns heatmap
            st.subheader("Monthly Returns Heatmap")
            
            if 'equity_curve' in results:
                # Calculate daily returns from equity curve
                daily_returns = results['equity_curve'].pct_change().dropna()
                
                # Create heatmap
                heatmap_fig = create_returns_heatmap(daily_returns)
                st.plotly_chart(heatmap_fig, use_container_width=True)
            
            # Underwater plot (drawdowns over time)
            st.subheader("Drawdown Analysis")
            
            if 'drawdowns' in results:
                underwater_fig = create_performance_chart(
                    None, 
                    results['drawdowns'],
                    chart_type='drawdown'
                )
                st.plotly_chart(underwater_fig, use_container_width=True)
            
            # Rolling statistics
            st.subheader("Rolling Performance Metrics")
            
            if 'equity_curve' in results:
                # Calculate rolling returns and volatility
                rolling_returns = results['equity_curve'].pct_change().dropna()
                
                # Calculate rolling annualized return
                rolling_return_30d = rolling_returns.rolling(30).mean() * 252 * 100  # Annualized, in percent
                
                # Calculate rolling volatility
                rolling_vol_30d = rolling_returns.rolling(30).std() * np.sqrt(252) * 100  # Annualized, in percent
                
                # Calculate rolling Sharpe ratio
                rolling_sharpe_30d = rolling_return_30d / rolling_vol_30d
                
                # Create DataFrame with rolling metrics
                rolling_df = pd.DataFrame({
                    'Rolling_Return_30d': rolling_return_30d,
                    'Rolling_Volatility_30d': rolling_vol_30d,
                    'Rolling_Sharpe_30d': rolling_sharpe_30d
                })
                
                # Plot rolling returns
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=rolling_df.index,
                    y=rolling_df['Rolling_Return_30d'],
                    name='30-Day Rolling Return (Annualized)',
                    line=dict(color='green', width=1)
                ))
                
                fig.add_trace(go.Scatter(
                    x=rolling_df.index,
                    y=rolling_df['Rolling_Volatility_30d'],
                    name='30-Day Rolling Volatility (Annualized)',
                    line=dict(color='red', width=1)
                ))
                
                fig.add_shape(
                    type="line",
                    x0=rolling_df.index[0],
                    y0=0,
                    x1=rolling_df.index[-1],
                    y1=0,
                    line=dict(color="gray", width=1, dash="dash"),
                )
                
                fig.update_layout(
                    title='Rolling Returns and Volatility (30-Day Window)',
                    xaxis_title='Date',
                    yaxis_title='Percent (%)',
                    template='plotly_dark',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Plot rolling Sharpe ratio
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=rolling_df.index,
                    y=rolling_df['Rolling_Sharpe_30d'],
                    name='30-Day Rolling Sharpe Ratio',
                    line=dict(color='blue', width=1)
                ))
                
                fig.add_shape(
                    type="line",
                    x0=rolling_df.index[0],
                    y0=1,
                    x1=rolling_df.index[-1],
                    y1=1,
                    line=dict(color="gray", width=1, dash="dash"),
                )
                
                fig.update_layout(
                    title='Rolling Sharpe Ratio (30-Day Window)',
                    xaxis_title='Date',
                    yaxis_title='Sharpe Ratio',
                    template='plotly_dark',
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)