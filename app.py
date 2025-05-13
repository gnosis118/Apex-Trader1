import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from data_processor import DataProcessor
from technical_analysis import TechnicalAnalysis
from backtester import Backtester
from rl_model import RLModel
from risk_management import RiskManager
from visualization import create_candlestick_chart, create_technical_indicator_chart, create_performance_chart

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
        ['Dashboard', 'Data Analysis', 'Backtesting', 'ML Model', 'Performance']
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
        ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        index=6
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
                st.session_state.data_processor = DataProcessor(symbol, timeframe, start_date, end_date)
                st.session_state.market_data = st.session_state.data_processor.get_data()
                
                # Initialize technical analysis
                st.session_state.ta = TechnicalAnalysis(st.session_state.market_data)
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
        st.session_state.position_size = st.slider("Position Size (%)", 1, 100, 10)
        st.session_state.stop_loss = st.slider("Stop Loss (%)", 1, 20, 5)
        st.session_state.take_profit = st.slider("Take Profit (%)", 1, 50, 15)
        st.session_state.max_drawdown = st.slider("Max Drawdown (%)", 5, 50, 25)

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
    - Machine learning model training
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
        st.markdown("### Machine Learning")
        st.write("Reinforcement learning for optimal trading")
        
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
                returns = st.session_state.market_data['Close'].pct_change().dropna()
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
        
        returns = st.session_state.market_data['Close'].pct_change().dropna()
        
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
            if use_ma:
                st.slider("Fast MA Period", 5, 50, st.session_state.ma_fast, key="comb_fast_ma")
                st.slider("Slow MA Period", 20, 200, st.session_state.ma_slow, key="comb_slow_ma")
            if use_rsi:
                st.slider("RSI Period", 7, 30, st.session_state.rsi_period, key="comb_rsi_period")
                st.slider("RSI Overbought", 60, 90, st.session_state.rsi_overbought, key="comb_rsi_overbought")
                st.slider("RSI Oversold", 10, 40, st.session_state.rsi_oversold, key="comb_rsi_oversold")
            if use_bb:
                st.slider("BB Period", 10, 50, st.session_state.bb_period, key="comb_bb_period")
                st.slider("BB Standard Deviation", 1.0, 3.0, st.session_state.bb_std, key="comb_bb_std")
            if use_macd:
                st.slider("MACD Fast", 8, 20, 12, key="comb_macd_fast")
                st.slider("MACD Slow", 20, 40, 26, key="comb_macd_slow")
                st.slider("MACD Signal", 5, 15, 9, key="comb_macd_signal")
        
        # Risk management
        st.subheader("Risk Management Parameters")
        position_size = st.slider("Position Size (%)", 1, 100, st.session_state.position_size, key="bt_position_size")
        stop_loss = st.slider("Stop Loss (%)", 1, 20, st.session_state.stop_loss, key="bt_stop_loss")
        take_profit = st.slider("Take Profit (%)", 1, 50, st.session_state.take_profit, key="bt_take_profit")
        max_drawdown = st.slider("Max Drawdown (%)", 5, 50, st.session_state.max_drawdown, key="bt_max_drawdown")
        
        # Run backtest button
        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                try:
                    # Initialize backtester
                    backtester = Backtester(st.session_state.market_data_with_indicators)
                    
                    # Run backtest based on selected strategy
                    if strategy_type == "Moving Average Crossover":
                        bt_results = backtester.run_ma_crossover_strategy(
                            fast_period=fast_ma,
                            slow_period=slow_ma,
                            position_size=position_size/100,
                            stop_loss=stop_loss/100,
                            take_profit=take_profit/100,
                            max_drawdown=max_drawdown/100
                        )
                    elif strategy_type == "RSI Mean Reversion":
                        bt_results = backtester.run_rsi_strategy(
                            rsi_period=rsi_period,
                            overbought=rsi_overbought,
                            oversold=rsi_oversold,
                            position_size=position_size/100,
                            stop_loss=stop_loss/100,
                            take_profit=take_profit/100,
                            max_drawdown=max_drawdown/100
                        )
                    elif strategy_type == "Bollinger Bands":
                        bt_results = backtester.run_bollinger_bands_strategy(
                            bb_period=bb_period,
                            bb_std=bb_std,
                            position_size=position_size/100,
                            stop_loss=stop_loss/100,
                            take_profit=take_profit/100,
                            max_drawdown=max_drawdown/100
                        )
                    elif strategy_type == "MACD":
                        bt_results = backtester.run_macd_strategy(
                            fast_period=macd_fast,
                            slow_period=macd_slow,
                            signal_period=macd_signal,
                            position_size=position_size/100,
                            stop_loss=stop_loss/100,
                            take_profit=take_profit/100,
                            max_drawdown=max_drawdown/100
                        )
                    elif strategy_type == "Combined Strategy":
                        # Create a list of indicators to use
                        indicators = []
                        params = {}
                        
                        if use_ma:
                            indicators.append("ma_crossover")
                            params["fast_ma"] = st.session_state.comb_fast_ma
                            params["slow_ma"] = st.session_state.comb_slow_ma
                        if use_rsi:
                            indicators.append("rsi")
                            params["rsi_period"] = st.session_state.comb_rsi_period
                            params["rsi_overbought"] = st.session_state.comb_rsi_overbought
                            params["rsi_oversold"] = st.session_state.comb_rsi_oversold
                        if use_bb:
                            indicators.append("bollinger_bands")
                            params["bb_period"] = st.session_state.comb_bb_period
                            params["bb_std"] = st.session_state.comb_bb_std
                        if use_macd:
                            indicators.append("macd")
                            params["macd_fast"] = st.session_state.comb_macd_fast
                            params["macd_slow"] = st.session_state.comb_macd_slow
                            params["macd_signal"] = st.session_state.comb_macd_signal
                            
                        bt_results = backtester.run_combined_strategy(
                            indicators=indicators,
                            params=params,
                            position_size=position_size/100,
                            stop_loss=stop_loss/100,
                            take_profit=take_profit/100,
                            max_drawdown=max_drawdown/100
                        )
                    
                    st.session_state.backtest_results = bt_results
                    st.success("Backtest completed successfully!")
                    
                except Exception as e:
                    st.error(f"Error during backtesting: {str(e)}")
        
        # Show backtest results if available
        if st.session_state.backtest_results is not None:
            bt_results = st.session_state.backtest_results
            
            # Performance metrics
            st.subheader("Backtest Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{bt_results['total_return']:.2f}%")
                st.metric("Sharpe Ratio", f"{bt_results['sharpe_ratio']:.2f}")
            with col2:
                st.metric("Max Drawdown", f"{bt_results['max_drawdown']:.2f}%")
                st.metric("Win Rate", f"{bt_results['win_rate']:.2f}%")
            with col3:
                st.metric("Profit Factor", f"{bt_results['profit_factor']:.2f}")
                st.metric("Avg Win", f"{bt_results['avg_win']:.2f}%")
            with col4:
                st.metric("Avg Loss", f"{bt_results['avg_loss']:.2f}%")
                st.metric("Total Trades", f"{bt_results['total_trades']}")
            
            # Equity curve
            st.subheader("Equity Curve")
            eq_fig = create_performance_chart(bt_results['equity_curve'], chart_type='equity')
            st.plotly_chart(eq_fig, use_container_width=True)
            
            # Trades visualization
            st.subheader("Trade Visualization")
            
            # Show the price chart with buy/sell markers
            trades_fig = create_candlestick_chart(
                st.session_state.market_data_with_indicators,
                ma_fast=st.session_state.ma_fast,
                ma_slow=st.session_state.ma_slow,
                bb_period=st.session_state.bb_period,
                bb_std=st.session_state.bb_std,
                buy_signals=bt_results['buy_signals'] if 'buy_signals' in bt_results else None,
                sell_signals=bt_results['sell_signals'] if 'sell_signals' in bt_results else None
            )
            st.plotly_chart(trades_fig, use_container_width=True)
            
            # Trade details table
            st.subheader("Trade Details")
            
            if 'trades' in bt_results:
                trade_df = pd.DataFrame(bt_results['trades'])
                st.dataframe(trade_df, use_container_width=True)
    
    # ML Model Tab
    elif st.session_state.current_tab == 'ML Model':
        st.header("Machine Learning Model")
        
        # Model parameters
        st.subheader("Reinforcement Learning Parameters")
        
        # Model selection
        model_type = st.selectbox(
            "Model Type",
            ["A2C", "PPO", "DQN"],
            index=1
        )
        
        # Model hyperparameters
        learning_rate = st.slider("Learning Rate", 1e-5, 1e-2, 1e-4, format="%.5f")
        gamma = st.slider("Discount Factor (Gamma)", 0.9, 0.999, 0.99, format="%.3f")
        
        # Training parameters
        col1, col2 = st.columns(2)
        with col1:
            training_episodes = st.number_input("Training Episodes", 10, 10000, 1000, 100)
        with col2:
            validation_split = st.slider("Validation Split", 0.1, 0.5, 0.2, 0.05)
        
        # State space configuration
        st.subheader("State Space Configuration")
        
        # Let user select features to include in the state space
        st.write("Select features to include in the state space:")
        use_price = st.checkbox("Price Data", True)
        use_volume = st.checkbox("Volume Data", True)
        use_ma = st.checkbox("Moving Averages", True)
        use_rsi = st.checkbox("RSI", True)
        use_bb = st.checkbox("Bollinger Bands", True)
        use_macd = st.checkbox("MACD", True)
        
        # Action space configuration
        st.subheader("Action Space Configuration")
        action_space = st.radio(
            "Action Space",
            ["Discrete (Buy, Sell, Hold)", "Continuous (Position Sizing)"]
        )
        
        # Reward function configuration
        st.subheader("Reward Function Configuration")
        reward_function = st.selectbox(
            "Reward Function",
            ["Pure Return", "Return - Risk Penalty", "Sharpe Ratio", "Custom"]
        )
        
        if reward_function == "Custom":
            return_weight = st.slider("Return Weight", 0.0, 1.0, 0.7, 0.1)
            risk_weight = st.slider("Risk Weight", 0.0, 1.0, 0.3, 0.1)
            consistency_weight = st.slider("Consistency Weight", 0.0, 1.0, 0.2, 0.1)
        
        # Train model button
        if st.button("Train Model"):
            with st.spinner("Training model... This may take a while"):
                try:
                    # Create feature list
                    features = []
                    if use_price:
                        features.extend(['Open', 'High', 'Low', 'Close'])
                    if use_volume:
                        features.append('Volume')
                    if use_ma:
                        features.extend([f'MA_{st.session_state.ma_fast}', f'MA_{st.session_state.ma_slow}'])
                    if use_rsi:
                        features.append(f'RSI_{st.session_state.rsi_period}')
                    if use_bb:
                        features.extend([f'BB_Upper_{st.session_state.bb_period}', f'BB_Lower_{st.session_state.bb_period}', 
                                        f'BB_Middle_{st.session_state.bb_period}'])
                    if use_macd:
                        features.extend(['MACD', 'MACD_Signal', 'MACD_Hist'])
                    
                    # Create reward function config
                    if reward_function == "Pure Return":
                        reward_config = {"type": "return"}
                    elif reward_function == "Return - Risk Penalty":
                        reward_config = {"type": "return_risk", "risk_factor": 0.5}
                    elif reward_function == "Sharpe Ratio":
                        reward_config = {"type": "sharpe"}
                    elif reward_function == "Custom":
                        reward_config = {
                            "type": "custom",
                            "return_weight": return_weight,
                            "risk_weight": risk_weight,
                            "consistency_weight": consistency_weight
                        }
                    
                    # Initialize RL model
                    rl_model = RLModel(
                        data=st.session_state.market_data_with_indicators,
                        features=features,
                        action_type="discrete" if action_space == "Discrete (Buy, Sell, Hold)" else "continuous",
                        model_type=model_type.lower(),
                        learning_rate=learning_rate,
                        gamma=gamma,
                        reward_config=reward_config
                    )
                    
                    # Train the model
                    training_results = rl_model.train(
                        episodes=training_episodes,
                        validation_split=validation_split
                    )
                    
                    # Store training results in session state
                    st.session_state.model_trained = True
                    st.session_state.rl_model = rl_model
                    st.session_state.training_results = training_results
                    
                    st.success("Model training completed!")
                    
                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")
        
        # Show training results if model is trained
        if st.session_state.model_trained and 'training_results' in st.session_state:
            training_results = st.session_state.training_results
            
            # Training metrics
            st.subheader("Training Metrics")
            
            # Plot training rewards
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=training_results['episode_rewards'],
                mode='lines',
                name='Rewards',
                line=dict(color='blue')
            ))
            fig.update_layout(
                title='Episode Rewards During Training',
                xaxis_title='Episode',
                yaxis_title='Total Reward',
                template='plotly_dark'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Model evaluation
            st.subheader("Model Evaluation")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Final Reward", f"{training_results['final_reward']:.2f}")
            with col2:
                st.metric("Mean Reward", f"{training_results['mean_reward']:.2f}")
            with col3:
                st.metric("Max Reward", f"{training_results['max_reward']:.2f}")
            with col4:
                st.metric("Training Time", f"{training_results['training_time']:.2f} s")
            
            # Performance on validation set
            st.subheader("Performance on Validation Set")
            
            # Plot model performance on validation data
            if 'validation_equity_curve' in training_results:
                val_fig = create_performance_chart(
                    training_results['validation_equity_curve'],
                    chart_type='equity'
                )
                st.plotly_chart(val_fig, use_container_width=True)
            
            # Model actions visualization
            st.subheader("Model Actions Visualization")
            
            if 'model_actions' in training_results:
                # Create figure with price data and model actions
                actions_fig = create_candlestick_chart(
                    st.session_state.market_data_with_indicators.iloc[-100:],  # Last 100 data points
                    ma_fast=st.session_state.ma_fast,
                    ma_slow=st.session_state.ma_slow,
                    buy_signals=training_results['model_actions']['buy'] if 'buy' in training_results['model_actions'] else None,
                    sell_signals=training_results['model_actions']['sell'] if 'sell' in training_results['model_actions'] else None
                )
                st.plotly_chart(actions_fig, use_container_width=True)
    
    # Performance Tab
    elif st.session_state.current_tab == 'Performance':
        st.header("Performance Analytics")
        
        # Check if we have backtest results
        if st.session_state.backtest_results is None:
            st.warning("No backtest results available. Please run a backtest in the Backtesting tab first.")
        else:
            bt_results = st.session_state.backtest_results
            
            # Overall performance metrics
            st.subheader("Overall Performance")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Return", f"{bt_results['total_return']:.2f}%")
                st.metric("Annual Return", f"{bt_results['annual_return']:.2f}%")
                st.metric("Volatility", f"{bt_results['volatility']:.2f}%")
            with col2:
                st.metric("Sharpe Ratio", f"{bt_results['sharpe_ratio']:.2f}")
                st.metric("Sortino Ratio", f"{bt_results['sortino_ratio']:.2f}")
                st.metric("Calmar Ratio", f"{bt_results['calmar_ratio']:.2f}")
            with col3:
                st.metric("Max Drawdown", f"{bt_results['max_drawdown']:.2f}%")
                st.metric("Win Rate", f"{bt_results['win_rate']:.2f}%")
                st.metric("Profit Factor", f"{bt_results['profit_factor']:.2f}")
            
            # Equity curve with drawdowns
            st.subheader("Equity Curve with Drawdowns")
            equity_dd_fig = create_performance_chart(
                bt_results['equity_curve'],
                drawdowns=bt_results['drawdowns'] if 'drawdowns' in bt_results else None,
                chart_type='equity_drawdown'
            )
            st.plotly_chart(equity_dd_fig, use_container_width=True)
            
            # Monthly returns heatmap
            st.subheader("Monthly Returns Heatmap")
            
            if 'monthly_returns' in bt_results:
                # Create heatmap of monthly returns
                monthly_returns = bt_results['monthly_returns']
                
                # Create the heatmap figure
                fig = go.Figure(data=go.Heatmap(
                    z=monthly_returns.values,
                    x=monthly_returns.columns,
                    y=monthly_returns.index,
                    colorscale='RdYlGn',
                    zmid=0
                ))
                fig.update_layout(
                    title='Monthly Returns (%)',
                    template='plotly_dark'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Risk analysis
            st.subheader("Risk Analysis")
            
            # Value at Risk (VaR) and Expected Shortfall
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Daily VaR (95%)", f"{bt_results['var_95']:.2f}%")
                st.metric("Daily VaR (99%)", f"{bt_results['var_99']:.2f}%")
            with col2:
                st.metric("Expected Shortfall (95%)", f"{bt_results['es_95']:.2f}%")
                st.metric("Expected Shortfall (99%)", f"{bt_results['es_99']:.2f}%")
            
            # Return distribution
            st.subheader("Return Distribution")
            
            if 'returns' in bt_results:
                # Create histogram of returns
                returns = bt_results['returns']
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=returns,
                    nbinsx=50,
                    name='Returns',
                    marker_color='blue'
                ))
                
                # Add normal distribution for comparison
                import scipy.stats as stats
                import numpy as np
                
                mu, sigma = returns.mean(), returns.std()
                x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
                y = stats.norm.pdf(x, mu, sigma)
                
                # Scale the PDF to match the histogram
                y = y * len(returns) * (returns.max() - returns.min()) / 50
                
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    name='Normal Distribution',
                    line=dict(color='red')
                ))
                
                fig.update_layout(
                    title='Return Distribution vs. Normal Distribution',
                    xaxis_title='Daily Returns',
                    yaxis_title='Frequency',
                    template='plotly_dark'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Trade analysis
            st.subheader("Trade Analysis")
            
            if 'trades' in bt_results:
                trades = pd.DataFrame(bt_results['trades'])
                
                # Trade duration histogram
                if 'duration' in trades.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=trades['duration'],
                        nbinsx=20,
                        marker_color='blue'
                    ))
                    fig.update_layout(
                        title='Trade Duration Distribution',
                        xaxis_title='Duration (days)',
                        yaxis_title='Number of Trades',
                        template='plotly_dark'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Profit/loss by trade
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=trades.index,
                    y=trades['profit_pct'],
                    marker_color=['green' if profit >= 0 else 'red' for profit in trades['profit_pct']]
                ))
                fig.update_layout(
                    title='Profit/Loss by Trade',
                    xaxis_title='Trade Number',
                    yaxis_title='Profit/Loss (%)',
                    template='plotly_dark'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Trade statistics
                st.subheader("Trade Statistics")
                
                # Calculate additional trade statistics
                avg_win = trades[trades['profit_pct'] > 0]['profit_pct'].mean() if len(trades[trades['profit_pct'] > 0]) > 0 else 0
                avg_loss = trades[trades['profit_pct'] < 0]['profit_pct'].mean() if len(trades[trades['profit_pct'] < 0]) > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Trades", f"{len(trades)}")
                    st.metric("Winning Trades", f"{len(trades[trades['profit_pct'] > 0])}")
                with col2:
                    st.metric("Losing Trades", f"{len(trades[trades['profit_pct'] < 0])}")
                    st.metric("Win Rate", f"{bt_results['win_rate']:.2f}%")
                with col3:
                    st.metric("Avg Win", f"{avg_win:.2f}%")
                    st.metric("Avg Loss", f"{avg_loss:.2f}%")
                with col4:
                    st.metric("Best Trade", f"{trades['profit_pct'].max():.2f}%")
                    st.metric("Worst Trade", f"{trades['profit_pct'].min():.2f}%")
                
                # Display the trade data
                st.subheader("Trade Data")
                st.dataframe(trades, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Self-Learning Futures Trading Bot | Developed with Streamlit")
