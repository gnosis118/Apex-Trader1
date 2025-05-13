import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

# Import database module (simple version)
import db_simple as database

# Set page configuration
st.set_page_config(
    page_title="Futures Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
try:
    if database.check_connection():
        database.initialize_database()
        st.sidebar.success("Database connected", icon="✅")
    else:
        st.sidebar.error("Database connection failed")
except Exception as e:
    st.sidebar.error(f"Database error: {str(e)}")

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
        ['Dashboard', 'Data Analysis', 'Backtesting', 'Performance', 'Database']
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
    data_source = st.radio("Data Source", ["Sample Data", "Database", "Live Data"])
    
    # Load data button
    if st.button("Load Data"):
        with st.spinner("Loading market data..."):
            try:
                if data_source == "Sample Data":
                    # Use sample data
                    st.session_state.market_data = st.session_state.sample_data
                    st.session_state.data_loaded = True
                    st.success(f"Successfully loaded sample data")
                elif data_source == "Database":
                    # Try to load from database
                    try:
                        start_datetime = datetime.combine(start_date, datetime.min.time())
                        end_datetime = datetime.combine(end_date, datetime.max.time())
                        db_data = database.load_market_data(symbol, timeframe, start_datetime, end_datetime)
                        
                        if db_data.empty:
                            st.warning("No data found in database. Loading sample data instead.")
                            st.session_state.market_data = st.session_state.sample_data
                        else:
                            st.session_state.market_data = db_data
                            
                            # Add technical indicators
                            # Add moving averages
                            st.session_state.market_data['MA_20'] = st.session_state.market_data['Close'].rolling(window=20).mean()
                            st.session_state.market_data['MA_50'] = st.session_state.market_data['Close'].rolling(window=50).mean()
                            st.session_state.market_data['MA_200'] = st.session_state.market_data['Close'].rolling(window=200).mean()
                            
                            # Add RSI
                            delta = st.session_state.market_data['Close'].diff()
                            gain = delta.where(delta > 0, 0)
                            loss = -delta.where(delta < 0, 0)
                            avg_gain = gain.rolling(window=14).mean()
                            avg_loss = loss.rolling(window=14).mean()
                            rs = avg_gain / avg_loss
                            st.session_state.market_data['RSI'] = 100 - (100 / (1 + rs))
                            
                            # Add Returns
                            st.session_state.market_data['Returns'] = st.session_state.market_data['Close'].pct_change()
                            
                        st.session_state.data_loaded = True
                        st.success(f"Successfully loaded data from database")
                    except Exception as e:
                        st.error(f"Error loading from database: {str(e)}")
                        # Fall back to sample data
                        st.session_state.market_data = st.session_state.sample_data
                        st.session_state.data_loaded = True
                else:
                    # Display a message about live data being unavailable
                    st.error("Live data connection requires additional dependencies. Falling back to sample data.")
                    st.session_state.market_data = st.session_state.sample_data
                    st.session_state.data_loaded = True
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    # Save data to database button (only available when data is loaded)
    if st.session_state.data_loaded:
        if st.button("Save Data to Database"):
            with st.spinner("Saving market data to database..."):
                try:
                    database.save_market_data(st.session_state.market_data, symbol, timeframe)
                    st.success("Data saved to database successfully")
                except Exception as e:
                    st.error(f"Error saving data: {str(e)}")
    
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
    - Database storage for market data and strategies
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
                fast_ma = st.slider("Fast MA Period", 5, 50, st.session_state.ma_fast, key="comb_fast_ma")
                slow_ma = st.slider("Slow MA Period", 20, 200, st.session_state.ma_slow, key="comb_slow_ma")
            else:
                fast_ma = st.session_state.ma_fast
                slow_ma = st.session_state.ma_slow
            
            if use_rsi:
                rsi_period = st.slider("RSI Period", 7, 30, st.session_state.rsi_period, key="comb_rsi_period")
                rsi_overbought = st.slider("RSI Overbought", 60, 90, st.session_state.rsi_overbought, key="comb_rsi_overbought")
                rsi_oversold = st.slider("RSI Oversold", 10, 40, st.session_state.rsi_oversold, key="comb_rsi_oversold")
            else:
                rsi_period = st.session_state.rsi_period
                rsi_overbought = st.session_state.rsi_overbought
                rsi_oversold = st.session_state.rsi_oversold
        
        # Strategy name input for saving
        strategy_name = st.text_input("Strategy Name", f"{strategy_type} Strategy")
        strategy_description = st.text_area("Strategy Description", "")
        
        # Run backtest button
        run_col, save_col = st.columns(2)
        with run_col:
            if st.button("Run Backtest"):
                st.info("Running backtest...")
                
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
                    ma_signal = np.zeros(len(data))
                    rsi_signal = np.zeros(len(data))
                    
                    if use_ma:
                        ma_signal[(data['MA_20'] > data['MA_50'])] = 1
                        ma_signal[(data['MA_20'] < data['MA_50'])] = -1
                    
                    if use_rsi:
                        rsi_signal[data['RSI'] < rsi_oversold] = 1
                        rsi_signal[data['RSI'] > rsi_overbought] = -1
                    
                    if use_ma and use_rsi:
                        # Combined signal
                        data['Signal'] = np.where(ma_signal == rsi_signal, ma_signal, 0)
                    elif use_ma:
                        data['Signal'] = ma_signal
                    elif use_rsi:
                        data['Signal'] = rsi_signal
                
                # Calculate strategy returns
                data['Position'] = data['Signal'].shift(1).fillna(0)
                data['Strategy_Return'] = data['Position'] * data['Returns']
                data['Cumulative_Return'] = (1 + data['Strategy_Return']).cumprod()
                
                # Store in session state
                st.session_state.backtest_results = data
                
                # Calculate performance metrics
                returns = data['Strategy_Return'].dropna()
                
                # Win rate
                wins = sum(returns > 0)
                losses = sum(returns < 0)
                win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
                
                # Profit factor
                total_wins = sum(returns[returns > 0])
                total_losses = abs(sum(returns[returns < 0]))
                profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
                
                # Annualized return
                total_return = data['Cumulative_Return'].iloc[-1] - 1
                trading_days = len(returns)
                annualized_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0
                
                # Volatility
                volatility = returns.std() * np.sqrt(252)
                
                # Sharpe Ratio (assuming 0% risk-free rate)
                sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
                
                # Maximum Drawdown
                cumulative = data['Cumulative_Return']
                peak = cumulative.cummax()
                drawdown = (cumulative / peak - 1)
                max_drawdown = drawdown.min()
                
                # Extract trades
                trades = []
                position_changes = data['Position'].diff().fillna(0)
                position = 0
                entry_date = None
                entry_price = None
                
                for i in range(len(data)):
                    if position == 0 and data['Position'].iloc[i] != 0:
                        # Entry
                        position = data['Position'].iloc[i]
                        entry_date = data.index[i]
                        entry_price = data['Close'].iloc[i]
                    elif position != 0 and (data['Position'].iloc[i] == 0 or data['Position'].iloc[i] * position < 0):
                        # Exit
                        exit_date = data.index[i]
                        exit_price = data['Close'].iloc[i]
                        profit_pct = (exit_price / entry_price - 1) * position * 100  # Convert to percentage
                        
                        trade = {
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'direction': position,
                            'profit_pct': profit_pct,
                            'duration': (exit_date - entry_date).days
                        }
                        trades.append(trade)
                        
                        # Reset for next trade
                        position = data['Position'].iloc[i]
                        if position != 0:
                            entry_date = data.index[i]
                            entry_price = data['Close'].iloc[i]
                        else:
                            entry_date = None
                            entry_price = None
                
                # Create results dictionary
                results = {
                    'total_return': total_return * 100,  # Convert to percentage
                    'annualized_return': annualized_return * 100,  # Convert to percentage
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': abs(max_drawdown) * 100,  # Convert to percentage
                    'win_rate': win_rate * 100,  # Convert to percentage
                    'profit_factor': profit_factor,
                    'trades_count': len(trades),
                    'trades': trades
                }
                
                # Store in session state
                st.session_state.backtest_metrics = results
                
                st.success("Backtest completed!")
                
                # Display basic results
                st.subheader("Backtest Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Return", f"{results['total_return']:.2f}%")
                with col2:
                    st.metric("Annualized Return", f"{results['annualized_return']:.2f}%")
                with col3:
                    st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                with col4:
                    st.metric("Max Drawdown", f"{results['max_drawdown']:.2f}%")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Win Rate", f"{results['win_rate']:.2f}%")
                with col2:
                    st.metric("Profit Factor", f"{results['profit_factor']:.2f}")
                with col3:
                    st.metric("Number of Trades", results['trades_count'])
                
                # Show equity curve
                st.subheader("Equity Curve")
                st.line_chart(data['Cumulative_Return'])
        
        with save_col:
            # Only enable save button if backtest has been run
            if 'backtest_metrics' in st.session_state:
                if st.button("Save Strategy & Results to Database"):
                    try:
                        # Collect strategy parameters
                        if strategy_type == "Moving Average Crossover":
                            parameters = {
                                'fast_ma': fast_ma,
                                'slow_ma': slow_ma,
                                'position_size': st.session_state.position_size,
                                'stop_loss': st.session_state.stop_loss,
                                'take_profit': st.session_state.take_profit,
                                'max_drawdown': st.session_state.max_drawdown
                            }
                        elif strategy_type == "RSI Mean Reversion":
                            parameters = {
                                'rsi_period': rsi_period,
                                'rsi_overbought': rsi_overbought,
                                'rsi_oversold': rsi_oversold,
                                'position_size': st.session_state.position_size,
                                'stop_loss': st.session_state.stop_loss,
                                'take_profit': st.session_state.take_profit,
                                'max_drawdown': st.session_state.max_drawdown
                            }
                        else:  # Combined Strategy
                            parameters = {
                                'use_ma': use_ma,
                                'fast_ma': fast_ma,
                                'slow_ma': slow_ma,
                                'use_rsi': use_rsi,
                                'rsi_period': rsi_period,
                                'rsi_overbought': rsi_overbought,
                                'rsi_oversold': rsi_oversold,
                                'position_size': st.session_state.position_size,
                                'stop_loss': st.session_state.stop_loss,
                                'take_profit': st.session_state.take_profit,
                                'max_drawdown': st.session_state.max_drawdown
                            }
                        
                        # Save strategy to database
                        strategy_id = database.save_strategy(
                            strategy_name,
                            strategy_type,
                            strategy_description,
                            parameters
                        )
                        
                        # Save backtest results
                        start_datetime = datetime.combine(start_date, datetime.min.time())
                        end_datetime = datetime.combine(end_date, datetime.max.time())
                        
                        backtest_id = database.save_backtest_result(
                            strategy_id,
                            symbol,
                            timeframe,
                            start_datetime,
                            end_datetime,
                            st.session_state.backtest_metrics
                        )
                        
                        st.success(f"Strategy and results saved to database (ID: {strategy_id})")
                    except Exception as e:
                        st.error(f"Error saving to database: {str(e)}")
    
    # Performance Tab
    elif st.session_state.current_tab == 'Performance':
        st.header("Performance Analytics")
        
        if 'backtest_results' not in st.session_state or st.session_state.backtest_results is None:
            # Try to load from database
            try:
                backtest_results = database.get_backtest_results()
                
                if not backtest_results:
                    st.info("No backtest results found in database. Run a backtest first.")
                else:
                    st.subheader("Saved Backtest Results")
                    
                    # Convert backtest results to DataFrame for display
                    results_df = pd.DataFrame(backtest_results)
                    
                    # Format columns
                    display_df = results_df.copy()
                    display_df['total_return'] = display_df['total_return'].map(lambda x: f"{x:.2f}%" if x is not None else 'N/A')
                    display_df['annualized_return'] = display_df['annualized_return'].map(lambda x: f"{x:.2f}%" if x is not None else 'N/A')
                    display_df['sharpe_ratio'] = display_df['sharpe_ratio'].map(lambda x: f"{x:.2f}" if x is not None else 'N/A')
                    display_df['max_drawdown'] = display_df['max_drawdown'].map(lambda x: f"{x:.2f}%" if x is not None else 'N/A')
                    display_df['win_rate'] = display_df['win_rate'].map(lambda x: f"{x:.2f}%" if x is not None else 'N/A')
                    
                    # Display as table
                    st.dataframe(
                        display_df[['strategy_name', 'symbol', 'timeframe', 'total_return', 
                                   'annualized_return', 'sharpe_ratio', 'max_drawdown', 
                                   'win_rate', 'trades_count', 'execution_date']], 
                        use_container_width=True
                    )
                    
                    # Allow selection of a specific result for more details
                    if not results_df.empty:
                        selected_backtest_id = st.selectbox(
                            "Select a backtest to view details:",
                            results_df['id'].tolist(),
                            format_func=lambda x: f"{results_df[results_df['id'] == x]['strategy_name'].iloc[0]} - {results_df[results_df['id'] == x]['symbol'].iloc[0]} ({results_df[results_df['id'] == x]['execution_date'].iloc[0]})"
                        )
                        
                        # Display the selected result
                        selected_result = results_df[results_df['id'] == selected_backtest_id].iloc[0]
                        
                        st.subheader(f"Details for {selected_result['strategy_name']}")
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Return", f"{selected_result['total_return']:.2f}%" if selected_result['total_return'] is not None else 'N/A')
                        with col2:
                            st.metric("Annualized Return", f"{selected_result['annualized_return']:.2f}%" if selected_result['annualized_return'] is not None else 'N/A')
                        with col3:
                            st.metric("Sharpe Ratio", f"{selected_result['sharpe_ratio']:.2f}" if selected_result['sharpe_ratio'] is not None else 'N/A')
                        with col4:
                            st.metric("Max Drawdown", f"{selected_result['max_drawdown']:.2f}%" if selected_result['max_drawdown'] is not None else 'N/A')
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Win Rate", f"{selected_result['win_rate']:.2f}%" if selected_result['win_rate'] is not None else 'N/A')
                        with col2:
                            st.metric("Profit Factor", f"{selected_result['profit_factor']:.2f}" if selected_result['profit_factor'] is not None else 'N/A')
                        with col3:
                            st.metric("Number of Trades", selected_result['trades_count'] if selected_result['trades_count'] is not None else 'N/A')
                        
                        # Note about equity curve not being available for database results
                        st.info("Detailed equity curve data is not stored in the database. Run a new backtest to see the equity curve.")
            except Exception as e:
                st.error(f"Error loading backtest results from database: {str(e)}")
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
    
    # Database Tab
    elif st.session_state.current_tab == 'Database':
        st.header("Database Management")
        
        tab1, tab2 = st.tabs(["Strategies", "Market Data"])
        
        with tab1:
            st.subheader("Saved Strategies")
            
            # Refresh button
            if st.button("Refresh Strategies"):
                st.rerun()
            
            # List all strategies
            try:
                strategies = database.get_strategies()
                
                if not strategies:
                    st.info("No strategies found in the database.")
                else:
                    # Convert to DataFrame for display
                    strategies_df = pd.DataFrame(strategies)
                    
                    # Format creation date
                    strategies_df['creation_date'] = pd.to_datetime(strategies_df['creation_date'])
                    
                    # Display strategies
                    st.dataframe(
                        strategies_df[['id', 'name', 'type', 'description', 'creation_date']], 
                        use_container_width=True
                    )
                    
                    # Allow selection of a specific strategy to view details
                    if not strategies_df.empty:
                        selected_strategy_id = st.selectbox(
                            "Select a strategy to view details:",
                            strategies_df['id'].tolist(),
                            format_func=lambda x: f"{strategies_df[strategies_df['id'] == x]['name'].iloc[0]}"
                        )
                        
                        # Display the selected strategy
                        selected_strategy = strategies_df[strategies_df['id'] == selected_strategy_id].iloc[0]
                        
                        st.subheader(f"Details for {selected_strategy['name']}")
                        
                        # Display parameters
                        st.write("**Type:** ", selected_strategy['type'])
                        st.write("**Description:** ", selected_strategy['description'])
                        st.write("**Created:** ", selected_strategy['creation_date'])
                        
                        st.write("**Parameters:**")
                        parameters = selected_strategy['parameters']
                        for key, value in parameters.items():
                            st.write(f"- {key}: {value}")
                        
                        # Display related backtest results
                        st.subheader("Backtest Results")
                        
                        try:
                            backtest_results = database.get_backtest_results(selected_strategy_id)
                            
                            if not backtest_results:
                                st.info("No backtest results found for this strategy.")
                            else:
                                # Convert to DataFrame
                                results_df = pd.DataFrame(backtest_results)
                                
                                # Format columns
                                results_df['total_return'] = results_df['total_return'].map(lambda x: f"{x:.2f}%" if x is not None else 'N/A')
                                results_df['annualized_return'] = results_df['annualized_return'].map(lambda x: f"{x:.2f}%" if x is not None else 'N/A')
                                results_df['sharpe_ratio'] = results_df['sharpe_ratio'].map(lambda x: f"{x:.2f}" if x is not None else 'N/A')
                                
                                # Display results
                                st.dataframe(
                                    results_df[['symbol', 'timeframe', 'total_return', 'annualized_return', 
                                               'sharpe_ratio', 'win_rate', 'trades_count', 'execution_date']], 
                                    use_container_width=True
                                )
                        except Exception as e:
                            st.error(f"Error loading backtest results: {str(e)}")
            except Exception as e:
                st.error(f"Error loading strategies from database: {str(e)}")
        
        with tab2:
            st.subheader("Market Data Management")
            
            # Data source form
            st.write("Query available market data:")
            
            col1, col2 = st.columns(2)
            with col1:
                db_symbol = st.selectbox(
                    "Select Symbol", 
                    ["ES=F", "NQ=F", "YM=F", "RTY=F", "CL=F", "GC=F", "SI=F", "ZC=F", "ZS=F", "6E=F", "6J=F", "BTC-USD"],
                    key="db_symbol"
                )
            with col2:
                db_timeframe = st.selectbox(
                    "Select Timeframe", 
                    ["1d", "4h", "1h", "30m", "15m", "5m", "1m"],
                    key="db_timeframe"
                )
            
            if st.button("Check Database"):
                try:
                    # Execute a simple query to check if the data exists
                    conn = database.get_connection()
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT COUNT(*) FROM market_data
                        WHERE symbol = %s AND timeframe = %s
                    """, (db_symbol, db_timeframe))
                    count = cursor.fetchone()[0]
                    cursor.close()
                    conn.close()
                    
                    if count > 0:
                        st.success(f"Found {count} records for {db_symbol} ({db_timeframe}) in the database")
                    else:
                        st.warning(f"No data found for {db_symbol} ({db_timeframe}) in the database")
                except Exception as e:
                    st.error(f"Error querying database: {str(e)}")
            
            # Database maintenance section
            st.subheader("Database Maintenance")
            
            maintenance_option = st.radio(
                "Maintenance Options",
                ["Information", "Initialize Database"]
            )
            
            if maintenance_option == "Information":
                st.info("""
                The database stores:
                - Market data for different symbols and timeframes
                - Strategy configurations
                - Backtest results and trades
                
                Data is stored in PostgreSQL and can be queried and analyzed.
                """)
            elif maintenance_option == "Initialize Database":
                if st.button("Initialize Database Tables"):
                    try:
                        database.initialize_database()
                        st.success("Database tables initialized successfully")
                    except Exception as e:
                        st.error(f"Error initializing database: {str(e)}")
                        
            st.markdown("""
            ---
            **Note:** This is a simple database interface. For more advanced operations,
            use the database management functions in the `db_simple.py` module.
            """)