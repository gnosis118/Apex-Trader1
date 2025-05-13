import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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
if 'sample_data' not in st.session_state:
    # Create a simple sample dataset for demonstration
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    close_prices = np.linspace(100, 150, len(dates)) + np.random.normal(0, 5, len(dates)).cumsum()
    volume = np.random.normal(1000000, 200000, len(dates))
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': close_prices * np.random.normal(0.99, 0.01, len(dates)),
        'High': close_prices * np.random.normal(1.02, 0.01, len(dates)),
        'Low': close_prices * np.random.normal(0.98, 0.01, len(dates)),
        'Close': close_prices,
        'Volume': volume
    })
    df = df.set_index('Date')
    
    # Add some basic indicators
    df['Returns'] = df['Close'].pct_change()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    
    # RSI calculation
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    st.session_state.sample_data = df

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
    
    # Data source selection
    data_source = st.radio("Data Source", ["Sample Data", "Live Data"])
    
    # Load data button
    if st.button("Load Data"):
        with st.spinner("Loading market data..."):
            try:
                if data_source == "Sample Data":
                    # Use sample data
                    st.session_state.market_data = st.session_state.sample_data
                    st.session_state.data_loaded = True
                    st.success(f"Successfully loaded sample data")
                else:
                    # Display a message about live data being unavailable
                    st.error("Live data connection requires additional dependencies. Please use sample data for now.")
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
    
    st.image("https://pixabay.com/get/gc73bb9e10face2db12906e9d4a3d077611cf0f39fe97ae7f212a436d875bd97d7790de2b7e0bfd6785e574afa98cae7978112475e6185e817d4d29b3c4f23689_1280.jpg", 
             caption="Financial Trading Dashboard")
    
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
    # Dashboard with data
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
            if 'RSI' in st.session_state.market_data.columns:
                rsi = st.session_state.market_data.iloc[-1]['RSI']
                st.metric("RSI", f"{rsi:.2f}", "")
            else:
                st.metric("RSI", "N/A", "")
                
        with metric4:
            if 'Returns' in st.session_state.market_data.columns:
                returns = st.session_state.market_data['Returns'].dropna()
                volatility = returns.std() * 100 * np.sqrt(252)  # Annualized volatility
                st.metric("Volatility", f"{volatility:.2f}%", "")
        
        # Main price chart
        st.subheader("Price Chart")
        
        # Create a simple line chart for the price (using Streamlit's native chart)
        st.line_chart(st.session_state.market_data[['Close']])
        
        # Show multiple technical indicators
        st.subheader("Technical Indicators")
        
        # Create tabs for different indicators
        tab1, tab2 = st.tabs(["Moving Averages", "RSI"])
        
        with tab1:
            # Moving Averages
            st.line_chart(st.session_state.market_data[['Close', 'MA_20', 'MA_50', 'MA_200']])
            
        with tab2:
            # RSI
            if 'RSI' in st.session_state.market_data.columns:
                st.line_chart(st.session_state.market_data[['RSI']])
                st.write("RSI Values: Above 70 is overbought, below 30 is oversold")
            
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
        
        # Plot returns histogram using Streamlit's native chart
        st.subheader("Returns Distribution")
        st.bar_chart(returns.value_counts(bins=20, normalize=True).sort_index())
        
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
        
        # Correlation matrix
        st.subheader("Correlation Matrix")
        corr = st.session_state.market_data[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
        st.dataframe(corr, use_container_width=True)
        
    # Backtesting Tab
    elif st.session_state.current_tab == 'Backtesting':
        st.header("Strategy Backtesting")
        
        # Strategy selection
        strategy_type = st.selectbox(
            "Select Strategy Type",
            ["Moving Average Crossover", "RSI Mean Reversion", "Combined Strategy"]
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
            
        elif strategy_type == "Combined Strategy":
            st.subheader("Combined Strategy Parameters")
            # Let user select which indicators to include
            use_ma = st.checkbox("Use Moving Averages", True)
            use_rsi = st.checkbox("Use RSI", True)
            
            # Only show parameters for selected indicators
            if use_ma:
                st.slider("Fast MA Period", 5, 50, st.session_state.ma_fast, key="comb_fast_ma")
                st.slider("Slow MA Period", 20, 200, st.session_state.ma_slow, key="comb_slow_ma")
            
            if use_rsi:
                st.slider("RSI Period", 7, 30, st.session_state.rsi_period, key="comb_rsi_period")
                st.slider("RSI Overbought", 60, 90, st.session_state.rsi_overbought, key="comb_rsi_overbought")
                st.slider("RSI Oversold", 10, 40, st.session_state.rsi_oversold, key="comb_rsi_oversold")
        
        # Run backtest button
        if st.button("Run Simple Backtest"):
            st.info("Running a simplified backtest for demonstration...")
            
            # Simple Moving Average Crossover strategy
            data = st.session_state.market_data.copy()
            
            # Create signals
            data['Signal'] = 0
            if strategy_type == "Moving Average Crossover":
                data.loc[data['MA_20'] > data['MA_50'], 'Signal'] = 1  # Buy signal
                data.loc[data['MA_20'] < data['MA_50'], 'Signal'] = -1  # Sell signal
            elif strategy_type == "RSI Mean Reversion":
                data.loc[data['RSI'] < 30, 'Signal'] = 1  # Buy when oversold
                data.loc[data['RSI'] > 70, 'Signal'] = -1  # Sell when overbought
            else:
                # Combined - simple version
                ma_signal = (data['MA_20'] > data['MA_50']).astype(int) * 2 - 1  # -1 or 1
                rsi_signal = np.zeros(len(data))
                rsi_signal[data['RSI'] < 30] = 1
                rsi_signal[data['RSI'] > 70] = -1
                data['Signal'] = np.where(ma_signal == rsi_signal, ma_signal, 0)
            
            # Calculate strategy returns
            data['Strategy_Return'] = data['Signal'].shift(1) * data['Returns']
            data['Cumulative_Return'] = (1 + data['Strategy_Return']).cumprod()
            
            # Store in session state
            st.session_state.backtest_results = data
            st.success("Backtest completed!")
            
            # Display basic results
            final_return = data['Cumulative_Return'].iloc[-1] - 1
            st.metric("Total Return", f"{final_return*100:.2f}%")
            
            # Show equity curve
            st.subheader("Equity Curve")
            st.line_chart(data['Cumulative_Return'])
    
    # Performance Tab
    elif st.session_state.current_tab == 'Performance':
        st.header("Performance Analytics")
        
        if 'backtest_results' not in st.session_state or st.session_state.backtest_results is None:
            st.info("Run a backtest first to see performance analytics.")
        else:
            data = st.session_state.backtest_results
            
            # Calculate performance metrics
            returns = data['Strategy_Return'].dropna()
            
            # Annualized return
            total_return = data['Cumulative_Return'].iloc[-1] - 1
            trading_days = len(returns)
            annualized_return = (1 + total_return) ** (252 / trading_days) - 1
            
            # Volatility
            volatility = returns.std() * np.sqrt(252)
            
            # Sharpe Ratio (assuming 0% risk-free rate)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Maximum Drawdown
            cumulative = data['Cumulative_Return']
            peak = cumulative.cummax()
            drawdown = (cumulative / peak - 1)
            max_drawdown = drawdown.min()
            
            # Win rate
            wins = sum(returns > 0)
            losses = sum(returns < 0)
            win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{total_return*100:.2f}%")
            with col2:
                st.metric("Annualized Return", f"{annualized_return*100:.2f}%")
            with col3:
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            with col4:
                st.metric("Max Drawdown", f"{abs(max_drawdown)*100:.2f}%")
                
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Win Rate", f"{win_rate*100:.2f}%")
            with col2:
                st.metric("Number of Trades", f"{wins + losses}")
            
            # Show drawdown chart
            st.subheader("Drawdown Over Time")
            st.line_chart(drawdown)
            
            # Show monthly returns
            st.subheader("Monthly Performance")
            
            # Convert index to datetime if needed
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
                
            # Calculate monthly returns
            monthly_returns = data['Strategy_Return'].resample('M').apply(lambda x: (1 + x).prod() - 1)
            
            # Display as a bar chart
            st.bar_chart(monthly_returns)