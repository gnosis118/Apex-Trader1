import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

# Import file storage module
import file_storage as storage
# Import reinforcement learning module
import simple_rl
# Import market regime analysis
import market_regime

# Set page configuration
st.set_page_config(
    page_title="Apex Trader",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize storage
try:
    if storage.check_connection():
        storage.initialize_storage()
        st.sidebar.success("Storage system initialized", icon="✅")
    else:
        st.sidebar.error("Storage system initialization failed")
except Exception as e:
    st.sidebar.error(f"Storage error: {str(e)}")

# Initialize session state variables
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
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
        ['Dashboard', 'Data Analysis', 'Market Regime', 'Backtesting', 'Training', 'Strategy Evolution', 'Live Trading', 'Performance', 'Storage']
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
    data_source = st.radio("Data Source", ["Sample Data", "File Storage", "Live Data"])
    
    # Load data button
    if st.button("Load Data"):
        with st.spinner("Loading market data..."):
            try:
                if data_source == "Sample Data":
                    # Use sample data
                    st.session_state.market_data = st.session_state.sample_data
                    st.session_state.data_loaded = True
                    st.success(f"Successfully loaded sample data")
                elif data_source == "File Storage":
                    # Try to load from file storage
                    try:
                        start_datetime = datetime.combine(start_date, datetime.min.time())
                        end_datetime = datetime.combine(end_date, datetime.max.time())
                        file_data = storage.load_market_data(symbol, timeframe, start_datetime, end_datetime)
                        
                        if file_data.empty:
                            st.warning("No data found in storage. Loading sample data instead.")
                            st.session_state.market_data = st.session_state.sample_data
                        else:
                            st.session_state.market_data = file_data
                            
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
                        st.success(f"Successfully loaded data from storage")
                    except Exception as e:
                        st.error(f"Error loading from storage: {str(e)}")
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
    
    # Save data to storage button (only available when data is loaded)
    if st.session_state.data_loaded:
        if st.button("Save Data to Storage"):
            with st.spinner("Saving market data to storage..."):
                try:
                    storage.save_market_data(st.session_state.market_data, symbol, timeframe)
                    st.success("Data saved to storage successfully")
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

# Add custom CSS styling
st.markdown("""
<style>
.main-header {
    color: #1A365D;
    font-weight: 600;
    margin-bottom: 0;
}
.tagline {
    color: #0047AB;
    font-style: italic;
    margin-top: 0;
    font-size: 1.2rem;
}
.profit-metric {
    color: #71EEB8;
}
.loss-metric {
    color: #FF6B6B;
}
.highlight-box {
    background-color: #F0F8FF;
    border-left: 5px solid #0047AB;
    padding: 10px;
    border-radius: 3px;
    margin-bottom: 20px;
}
.regime-trending-up {
    color: white;
    background-color: #71EEB8;
    padding: 4px 8px;
    border-radius: 3px;
}
.regime-trending-down {
    color: white;
    background-color: #FF6B6B;
    padding: 4px 8px;
    border-radius: 3px;
}
.regime-ranging {
    color: white;
    background-color: #0047AB;
    padding: 4px 8px;
    border-radius: 3px;
}
.regime-volatile {
    color: white;
    background-color: #FFA500;
    padding: 4px 8px;
    border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)

# Main content area based on selected tab
if not st.session_state.data_loaded:
    # Show sample dashboard with placeholders
    st.markdown("<h1 class='main-header'>APEX TRADER</h1>", unsafe_allow_html=True)
    st.markdown("<p class='tagline'>Apex fears no loss</p>", unsafe_allow_html=True)
    
    st.image("https://pixabay.com/get/gc73bb9e10face2db12906e9d4a3d077611cf0f39fe97ae7f212a436d875bd97d7790de2b7e0bfd6785e574afa98cae7978112475e6185e817d4d29b3c4f23689_1280.jpg", 
             caption="Advanced Trading Intelligence")
    
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
    - Reinforcement learning for autonomous trading
    - Performance analytics and risk management
    - Storage system for models, market data and strategies
    """)
    
    st.subheader("Key Features")
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    with feature_col1:
        st.markdown("### Technical Analysis")
        st.write("MACD, RSI, Bollinger Bands, Moving Averages")
    with feature_col2:
        st.markdown("### Self-Learning")
        st.write("Reinforcement learning for optimal trading decisions")
    with feature_col3:
        st.markdown("### Risk Management")
        st.write("Position sizing, stop-loss, and drawdown control")
        
else:
    # Dashboard with data
    if st.session_state.current_tab == 'Dashboard':
        st.markdown("<h1 class='main-header'>APEX TRADER DASHBOARD</h1>", unsafe_allow_html=True)
        st.markdown("<p class='tagline'>Apex fears no loss</p>", unsafe_allow_html=True)
        
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
        
        # RL Model Status
        st.subheader("AI Trading Status")
        
        model_status_col1, model_status_col2 = st.columns(2)
        
        with model_status_col1:
            if st.session_state.model_trained and st.session_state.current_model:
                st.success("✅ AI trading model is trained and ready")
                st.info(f"Current model: {st.session_state.current_model}")
            else:
                st.warning("⚠️ AI trading model not trained yet")
                st.info("Go to the 'Training' tab to train a model")
                
        with model_status_col2:
            if st.session_state.model_trained and st.session_state.current_model:
                # Model actions section
                st.subheader("Model Prediction")
                
                # Get model predictions
                try:
                    # Load the model
                    model_path = os.path.join(simple_rl.MODELS_DIR, st.session_state.current_model)
                    agent = simple_rl.TradingRLAgent.load_model(model_path)
                    
                    # Get current state
                    current_data = st.session_state.market_data.iloc[-10:].copy()
                    env = simple_rl.TradingEnvironment(current_data)
                    state = env._get_state()
                    
                    # Get action with exploration disabled
                    old_epsilon = agent.epsilon
                    agent.epsilon = 0
                    action = agent.act(state)
                    agent.epsilon = old_epsilon
                    
                    # Display action
                    if action == 0:
                        st.info("🔄 AI recommendation: HOLD")
                    elif action == 1:
                        st.success("🔼 AI recommendation: BUY")
                    else:
                        st.error("🔽 AI recommendation: SELL")
                except Exception as e:
                    st.error(f"Error getting model prediction: {str(e)}")
            
    # Data Analysis Tab    
    elif st.session_state.current_tab == 'Data Analysis':
        st.header("Data Analysis")
        
    # Market Regime Analysis Tab
    elif st.session_state.current_tab == 'Market Regime':
        st.header("Market Regime Analysis")
        
        if not st.session_state.data_loaded:
            st.warning("Please load market data first")
        else:
            # Initialize the market regime detector
            detector = market_regime.MarketRegimeDetector()
            
            # Analyze current market regime
            data = st.session_state.market_data.copy()
            
            # Detection parameters
            st.subheader("Detection Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                lookback_period = st.slider("Lookback Period", 10, 100, 50, 
                                          help="Number of periods to analyze for regime detection")
            
            with col2:
                period_type = st.selectbox("Analysis Period", 
                                          ["Recent", "Full Dataset", "Custom Range"],
                                          help="Timeframe to analyze")
            
            # Determine data range for analysis
            if period_type == "Recent":
                analysis_data = data.iloc[-lookback_period:] if len(data) > lookback_period else data
            elif period_type == "Custom Range":
                # Allow custom date range selection
                date_range = st.slider("Select Date Range",
                                      min_value=data.index.min().to_pydatetime(),
                                      max_value=data.index.max().to_pydatetime(),
                                      value=(data.index.min().to_pydatetime(), data.index.max().to_pydatetime()))
                
                start_date, end_date = date_range
                analysis_data = data.loc[start_date:end_date]
            else:  # Full Dataset
                analysis_data = data
            
            # Run the regime detection
            regime = detector.detect_regime(analysis_data, lookback_period=min(lookback_period, len(analysis_data)))
            
            # Display current regime
            st.subheader("Current Market Regime")
            
            # Display regime with color coding
            regime_color = {
                "TRENDING_UP": "green",
                "TRENDING_DOWN": "red",
                "RANGING": "blue",
                "VOLATILE": "orange",
                "MIXED": "gray",
                "UNKNOWN": "gray"
            }
            
            # Main regime status
            st.markdown(
                f"<h2 style='text-align: center; color: {regime_color.get(regime['regime'], 'white')}; "
                f"background-color: {regime_color.get(regime['regime'], 'gray')}30; "
                f"padding: 10px; border-radius: 5px;'>"
                f"{regime['regime']} (Confidence: {regime['confidence']:.2f})</h2>", 
                unsafe_allow_html=True
            )
            
            # Add regime transition analysis if we have sufficient history
            transition_analysis = detector.analyze_regime_transitions()
            
            if "prediction" not in transition_analysis:
                st.info(f"Predicted next regime: **{transition_analysis['predicted_next_regime']}** " 
                       f"(Probability: {transition_analysis['probability']:.2f})")
            
                # Display the transition matrix if it exists
                if st.checkbox("Show Regime Transition Analysis", False):
                    st.subheader("Regime Transition Probabilities")
                    
                    # Create a more readable format for the transition matrix
                    transition_data = []
                    for from_regime, to_regimes in transition_analysis.get('transition_matrix', {}).items():
                        for to_regime, probability in to_regimes.items():
                            transition_data.append({
                                "From Regime": from_regime,
                                "To Regime": to_regime,
                                "Probability": f"{probability:.2f}"
                            })
                    
                    if transition_data:
                        transitions_df = pd.DataFrame(transition_data)
                        st.dataframe(transitions_df)
            
            # Regime indicators
            st.subheader("Regime Indicators")
            
            # Create metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                trend_strength = regime["indicators"]["trend_strength"]
                st.metric("Trend Strength", 
                         f"{trend_strength:.2f}", 
                         delta=f"{trend_strength:.2f}")
                
            with col2:
                volatility = regime["indicators"]["volatility"]
                st.metric("Volatility", 
                         f"{volatility:.2f}", 
                         delta=f"{volatility:.2f}")
                
            with col3:
                ranging_vs_trending = regime["indicators"]["ranging_vs_trending"]
                st.metric("Range vs Trend", 
                         f"{ranging_vs_trending:.2f}", 
                         delta=f"{ranging_vs_trending:.2f}")
            
            # Recommended strategy
            st.subheader("Optimized Strategy Parameters")
            
            # Get optimal strategy for current regime
            optimal_strategy = detector.get_optimal_strategy(regime)
            
            # Display strategy parameters based on regime
            st.info(f"Based on the {regime['regime']} regime, the following strategy parameters are recommended:")
            
            # Create columns for strategy display
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Technical Indicators:**")
                st.write(f"- Use Moving Averages: {'Yes' if optimal_strategy.get('use_ma', True) else 'No'}")
                if optimal_strategy.get('use_ma', True):
                    st.write(f"- Fast MA Period: {optimal_strategy.get('fast_ma', 20)}")
                    st.write(f"- Slow MA Period: {optimal_strategy.get('slow_ma', 50)}")
                
                st.write(f"- Use RSI: {'Yes' if optimal_strategy.get('use_rsi', True) else 'No'}")
                if optimal_strategy.get('use_rsi', True):
                    st.write(f"- RSI Period: {optimal_strategy.get('rsi_period', 14)}")
                    st.write(f"- RSI Overbought: {optimal_strategy.get('rsi_overbought', 70)}")
                    st.write(f"- RSI Oversold: {optimal_strategy.get('rsi_oversold', 30)}")
            
            with col2:
                st.write("**Risk Management:**")
                st.write(f"- Position Size: {optimal_strategy.get('position_size', 0.1) * 100:.1f}%")
                st.write(f"- Stop Loss: {optimal_strategy.get('stop_loss', 0.05) * 100:.1f}%")
                st.write(f"- Take Profit: {optimal_strategy.get('take_profit', 0.1) * 100:.1f}%")
                st.write(f"- Max Drawdown: {optimal_strategy.get('max_drawdown', 0.25) * 100:.1f}%")
            
            # Apply recommended strategy to backtest
            if st.button("Apply to Backtesting"):
                # Store optimal strategy in session state
                st.session_state.optimal_strategy = optimal_strategy
                st.success("Strategy parameters applied! Go to Backtesting tab to run the optimized strategy.")
            
            # Market structure analysis
            st.subheader("Market Structure Analysis")
            
            # Add section for regime history
            with st.expander("Regime History", expanded=False):
                st.subheader("Historical Regime Changes")
                
                # Get regime history
                regime_history = detector.get_regime_history()
                
                if regime_history:
                    # Create a data frame for display
                    history_data = []
                    for entry in regime_history:
                        # Skip entries without timestamp (backward compatibility)
                        if 'timestamp' not in entry:
                            continue
                        
                        history_data.append({
                            "Timestamp": entry.get('timestamp', ''),
                            "Regime": entry.get('regime', 'UNKNOWN'),
                            "Confidence": f"{entry.get('confidence', 0.0):.2f}",
                            "Symbol": entry.get('symbol', 'unknown')
                        })
                    
                    if history_data:
                        history_df = pd.DataFrame(history_data)
                        st.dataframe(history_df)
                    else:
                        st.info("No recorded regime history with timestamps available yet.")
                else:
                    st.info("No regime history recorded yet.")
            
            # Analyze market structure
            market_structure = market_regime.analyze_market_structure(analysis_data, period=lookback_period)
            
            # Display support and resistance levels
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Support Levels:**")
                if market_structure["supports"]:
                    for level in market_structure["supports"]:
                        st.write(f"- {level:.2f}")
                else:
                    st.write("No significant support levels detected")
            
            with col2:
                st.write("**Resistance Levels:**")
                if market_structure["resistances"]:
                    for level in market_structure["resistances"]:
                        st.write(f"- {level:.2f}")
                else:
                    st.write("No significant resistance levels detected")
            
            # Price chart with levels
            st.subheader("Price Chart with Key Levels")
            
            # Create a price chart
            chart_data = analysis_data[['Close']].copy()
            
            # Add horizontal lines for key levels
            st.line_chart(chart_data)
            
            # Add a description based on the regime
            if regime["regime"] == "TRENDING_UP":
                st.success("""
                **Trading Strategy for Uptrend:**
                - Focus on trend-following strategies
                - Buy on pullbacks to key support levels or moving averages
                - Use wider stops to avoid getting shaken out of the trend
                - Look for continuation patterns
                """)
            elif regime["regime"] == "TRENDING_DOWN":
                st.error("""
                **Trading Strategy for Downtrend:**
                - Focus on trend-following strategies (short positions)
                - Sell on rallies to key resistance levels or moving averages
                - Use proper risk management as downtrends can be volatile
                - Be cautious of sudden reversals and policy interventions
                """)
            elif regime["regime"] == "RANGING":
                st.info("""
                **Trading Strategy for Range-Bound Market:**
                - Focus on mean-reversion strategies
                - Buy near support and sell near resistance
                - Use oscillators like RSI for overbought/oversold conditions
                - Be aware that ranges eventually break into trends
                """)
            elif regime["regime"] == "VOLATILE":
                st.warning("""
                **Trading Strategy for Volatile Market:**
                - Reduce position sizes
                - Widen stop losses to accommodate larger price swings
                - Expect false breakouts and whipsaws
                - Consider option strategies for directional uncertainty
                """)
            else:
                st.write("""
                **Trading Strategy for Mixed/Uncertain Market:**
                - Reduce exposure and position sizes
                - Wait for clearer signals before committing capital
                - Focus on shorter timeframes for more clarity
                - Monitor for regime change into a more definable pattern
                """)
        
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
        
        # Plot returns histogram
        st.subheader("Returns Distribution")
        
        # Use Streamlit's native histogram
        histogram_data = returns.value_counts(bins=20, normalize=True).sort_index()
        st.bar_chart(histogram_data)
        
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
        
        # Check if we have optimal strategy from Market Regime tab
        has_optimal_strategy = hasattr(st.session_state, 'optimal_strategy')
        
        if has_optimal_strategy:
            st.success("✅ Optimal strategy parameters from Market Regime analysis are available!")
            use_optimal = st.checkbox("Use Regime-Optimized Strategy", value=True)
        else:
            use_optimal = False
        
        # Strategy selection
        strategy_type = st.selectbox(
            "Select Strategy Type",
            ["Moving Average Crossover", "RSI Mean Reversion", "Combined Strategy", "AI Model (RL)"]
        )
        
        # Strategy parameters based on selection
        params = {}
        if strategy_type == "Moving Average Crossover":
            st.subheader("Moving Average Crossover Parameters")
            
            # If optimal strategy is available and user wants to use it
            if use_optimal and hasattr(st.session_state, 'optimal_strategy') and st.session_state.optimal_strategy.get('use_ma', True):
                # Use optimal parameters from regime detection
                optimal_strategy = st.session_state.optimal_strategy
                st.info("Using regime-optimized parameters")
                
                fast_ma = st.slider("Fast MA Period", 5, 50, 
                                   optimal_strategy.get('fast_ma', st.session_state.ma_fast), 
                                   key="bt_fast_ma")
                slow_ma = st.slider("Slow MA Period", 20, 200, 
                                   optimal_strategy.get('slow_ma', st.session_state.ma_slow), 
                                   key="bt_slow_ma")
                position_size = st.slider("Position Size (%)", 1, 100, 
                                         int(optimal_strategy.get('position_size', 0.1) * 100), 
                                         key="bt_position_size")
                stop_loss = st.slider("Stop Loss (%)", 1, 20, 
                                     int(optimal_strategy.get('stop_loss', 0.05) * 100), 
                                     key="bt_stop_loss")
                take_profit = st.slider("Take Profit (%)", 1, 30, 
                                       int(optimal_strategy.get('take_profit', 0.1) * 100), 
                                       key="bt_take_profit")
            else:
                # Use default parameters
                fast_ma = st.slider("Fast MA Period", 5, 50, st.session_state.ma_fast, key="bt_fast_ma")
                slow_ma = st.slider("Slow MA Period", 20, 200, st.session_state.ma_slow, key="bt_slow_ma")
                position_size = st.slider("Position Size (%)", 1, 100, 10, key="bt_position_size")
                stop_loss = st.slider("Stop Loss (%)", 1, 20, 5, key="bt_stop_loss")
                take_profit = st.slider("Take Profit (%)", 1, 30, 10, key="bt_take_profit")
            
            params["fast_ma"] = fast_ma
            params["slow_ma"] = slow_ma
            params["position_size"] = position_size / 100.0
            params["stop_loss"] = stop_loss / 100.0
            params["take_profit"] = take_profit / 100.0
            
        elif strategy_type == "RSI Mean Reversion":
            st.subheader("RSI Mean Reversion Parameters")
            
            # If optimal strategy is available and user wants to use it
            if use_optimal and hasattr(st.session_state, 'optimal_strategy') and st.session_state.optimal_strategy.get('use_rsi', True):
                # Use optimal parameters from regime detection
                optimal_strategy = st.session_state.optimal_strategy
                st.info("Using regime-optimized parameters")
                
                rsi_period = st.slider("RSI Period", 7, 30, 
                                      optimal_strategy.get('rsi_period', st.session_state.rsi_period), 
                                      key="bt_rsi_period")
                rsi_overbought = st.slider("Overbought Level", 60, 90, 
                                          optimal_strategy.get('rsi_overbought', st.session_state.rsi_overbought), 
                                          key="bt_rsi_overbought")
                rsi_oversold = st.slider("Oversold Level", 10, 40, 
                                        optimal_strategy.get('rsi_oversold', st.session_state.rsi_oversold), 
                                        key="bt_rsi_oversold")
                position_size = st.slider("Position Size (%)", 1, 100, 
                                         int(optimal_strategy.get('position_size', 0.1) * 100), 
                                         key="bt_rsi_position_size")
                stop_loss = st.slider("Stop Loss (%)", 1, 20, 
                                     int(optimal_strategy.get('stop_loss', 0.05) * 100), 
                                     key="bt_rsi_stop_loss")
                take_profit = st.slider("Take Profit (%)", 1, 30, 
                                       int(optimal_strategy.get('take_profit', 0.1) * 100), 
                                       key="bt_rsi_take_profit")
            else:
                # Use default parameters
                rsi_period = st.slider("RSI Period", 7, 30, st.session_state.rsi_period, key="bt_rsi_period")
                rsi_overbought = st.slider("Overbought Level", 60, 90, st.session_state.rsi_overbought, key="bt_rsi_overbought")
                rsi_oversold = st.slider("Oversold Level", 10, 40, st.session_state.rsi_oversold, key="bt_rsi_oversold")
                position_size = st.slider("Position Size (%)", 1, 100, 10, key="bt_rsi_position_size")
                stop_loss = st.slider("Stop Loss (%)", 1, 20, 5, key="bt_rsi_stop_loss")
                take_profit = st.slider("Take Profit (%)", 1, 30, 10, key="bt_rsi_take_profit")
            
            params["rsi_period"] = rsi_period
            params["rsi_overbought"] = rsi_overbought
            params["rsi_oversold"] = rsi_oversold
            params["position_size"] = position_size / 100.0
            params["stop_loss"] = stop_loss / 100.0
            params["take_profit"] = take_profit / 100.0
            
        elif strategy_type == "Combined Strategy":
            st.subheader("Combined Strategy Parameters")
            
            # If optimal strategy is available and user wants to use it
            if use_optimal and hasattr(st.session_state, 'optimal_strategy'):
                # Use optimal parameters from regime detection
                optimal_strategy = st.session_state.optimal_strategy
                st.info("Using regime-optimized parameters")
                
                # Let user select which indicators to include, with optimized defaults
                use_ma = st.checkbox("Use Moving Averages", optimal_strategy.get('use_ma', True))
                use_rsi = st.checkbox("Use RSI", optimal_strategy.get('use_rsi', True))
                
                # Risk parameters from optimal strategy
                position_size = st.slider("Position Size (%)", 1, 100, 
                                         int(optimal_strategy.get('position_size', 0.1) * 100), 
                                         key="comb_position_size")
                stop_loss = st.slider("Stop Loss (%)", 1, 20, 
                                     int(optimal_strategy.get('stop_loss', 0.05) * 100), 
                                     key="comb_stop_loss")
                take_profit = st.slider("Take Profit (%)", 1, 30, 
                                       int(optimal_strategy.get('take_profit', 0.1) * 100), 
                                       key="comb_take_profit")
            else:
                # Default indicator selection
                use_ma = st.checkbox("Use Moving Averages", True)
                use_rsi = st.checkbox("Use RSI", True)
                
                # Default risk parameters
                position_size = st.slider("Position Size (%)", 1, 100, 10, key="comb_position_size")
                stop_loss = st.slider("Stop Loss (%)", 1, 20, 5, key="comb_stop_loss")
                take_profit = st.slider("Take Profit (%)", 1, 30, 10, key="comb_take_profit")
            
            params["use_ma"] = use_ma
            params["use_rsi"] = use_rsi
            params["position_size"] = position_size / 100.0
            params["stop_loss"] = stop_loss / 100.0
            params["take_profit"] = take_profit / 100.0
            
            # Only show parameters for selected indicators
            if use_ma:
                if use_optimal and hasattr(st.session_state, 'optimal_strategy'):
                    fast_ma = st.slider("Fast MA Period", 5, 50, 
                                       optimal_strategy.get('fast_ma', st.session_state.ma_fast), 
                                       key="comb_fast_ma")
                    slow_ma = st.slider("Slow MA Period", 20, 200, 
                                       optimal_strategy.get('slow_ma', st.session_state.ma_slow), 
                                       key="comb_slow_ma")
                else:
                    fast_ma = st.slider("Fast MA Period", 5, 50, st.session_state.ma_fast, key="comb_fast_ma")
                    slow_ma = st.slider("Slow MA Period", 20, 200, st.session_state.ma_slow, key="comb_slow_ma")
                
                params["fast_ma"] = fast_ma
                params["slow_ma"] = slow_ma
            else:
                params["fast_ma"] = st.session_state.ma_fast
                params["slow_ma"] = st.session_state.ma_slow
            
            if use_rsi:
                if use_optimal and hasattr(st.session_state, 'optimal_strategy'):
                    rsi_period = st.slider("RSI Period", 7, 30, 
                                          optimal_strategy.get('rsi_period', st.session_state.rsi_period), 
                                          key="comb_rsi_period")
                    rsi_overbought = st.slider("RSI Overbought", 60, 90, 
                                              optimal_strategy.get('rsi_overbought', st.session_state.rsi_overbought), 
                                              key="comb_rsi_overbought")
                    rsi_oversold = st.slider("RSI Oversold", 10, 40, 
                                            optimal_strategy.get('rsi_oversold', st.session_state.rsi_oversold), 
                                            key="comb_rsi_oversold")
                else:
                    rsi_period = st.slider("RSI Period", 7, 30, st.session_state.rsi_period, key="comb_rsi_period")
                    rsi_overbought = st.slider("RSI Overbought", 60, 90, st.session_state.rsi_overbought, key="comb_rsi_overbought")
                    rsi_oversold = st.slider("RSI Oversold", 10, 40, st.session_state.rsi_oversold, key="comb_rsi_oversold")
                
                params["rsi_period"] = rsi_period
                params["rsi_overbought"] = rsi_overbought
                params["rsi_oversold"] = rsi_oversold
            else:
                params["rsi_period"] = st.session_state.rsi_period
                params["rsi_overbought"] = st.session_state.rsi_overbought
                params["rsi_oversold"] = st.session_state.rsi_oversold
                
        elif strategy_type == "AI Model (RL)":
            st.subheader("AI Model Selection")
            
            # Get available models
            available_models = simple_rl.get_all_models()
            
            if not available_models:
                st.warning("No trained AI models found. Go to the Training tab to train a model.")
            else:
                selected_model = st.selectbox("Select Model", available_models)
                st.write(f"Using reinforcement learning model: {selected_model}")
        
        # Risk management parameters
        params["position_size"] = st.session_state.position_size
        params["stop_loss"] = st.session_state.stop_loss
        params["take_profit"] = st.session_state.take_profit
        params["max_drawdown"] = st.session_state.max_drawdown
        
        # Strategy name input for saving
        strategy_name = st.text_input("Strategy Name", f"{strategy_type} Strategy")
        strategy_description = st.text_area("Strategy Description", "")
        
        # Option to use market regime optimized parameters
        use_regime_optimized = st.checkbox("Use Market Regime Optimized Parameters", False, 
                                         help="Automatically select strategy parameters based on the current market regime")
        
        # Run backtest button
        run_col, save_col = st.columns(2)
        with run_col:
            if st.button("Run Backtest"):
                with st.spinner("Running backtest..."):
                    # Check if we should use regime-optimized parameters
                    if use_regime_optimized:
                        try:
                            # Detect current market regime
                            detector = market_regime.MarketRegimeDetector()
                            selected_data = st.session_state.market_data.copy()
                            lookback = min(30, len(selected_data))
                            regime = detector.detect_regime(selected_data.iloc[-lookback:], lookback_period=lookback)
                            
                            st.info(f"Detected market regime: **{regime['regime']}** (Confidence: {regime['confidence']:.2f})")
                            
                            # Get optimal strategy parameters for this regime
                            optimal_params = detector.get_best_strategy(regime['regime'])
                            
                            if optimal_params and isinstance(optimal_params, dict):
                                # Extract parameters differently depending on the dict structure
                                if "parameters" in optimal_params:
                                    st.success(f"Using regime-optimized parameters for {regime['regime']}")
                                    # Update the parameters while preserving position sizing and risk mgmt settings
                                    strategy_params = optimal_params["parameters"]
                                    # Maintain risk settings from user input
                                    for key, value in strategy_params.items():
                                        if key not in ["position_size", "stop_loss", "take_profit", "max_drawdown"]:
                                            params[key] = value
                                else:
                                    st.success(f"Using regime-optimized parameters for {regime['regime']}")
                                    # Update the parameters while preserving position sizing and risk mgmt settings
                                    for key, value in optimal_params.items():
                                        if key not in ["position_size", "stop_loss", "take_profit", "max_drawdown"]:
                                            params[key] = value
                            else:
                                st.warning("No optimized parameters found for this regime. Using default parameters.")
                        except Exception as e:
                            st.error(f"Error determining optimal parameters: {str(e)}")
                            st.warning("Using default parameters instead.")
                    
                    # Initialize a control variable
                    continue_execution = True
                    
                    # Simple Moving Average Crossover strategy
                    data = st.session_state.market_data.copy()
                    
                    if strategy_type == "AI Model (RL)" and 'available_models' in locals() and available_models:
                        try:
                            # Load the model
                            model_path = os.path.join(simple_rl.MODELS_DIR, selected_model)
                            agent = simple_rl.TradingRLAgent.load_model(model_path)
                            
                            # Evaluate on data
                            eval_results = simple_rl.evaluate_agent(agent, data)
                            
                            # Create a results dictionary similar to other strategies
                            results = {
                                'total_return': eval_results['roi'],
                                'annualized_return': eval_results['roi'] / (len(data) / 252) * 100,
                                'sharpe_ratio': 0, # Not calculated for now
                                'max_drawdown': 0, # Not calculated for now
                                'win_rate': 0,     # Not calculated for now
                                'profit_factor': 0, # Not calculated for now
                                'trades_count': eval_results['trades'],
                                'trades': eval_results['trades_list']
                            }
                            
                            # Store in session state
                            st.session_state.backtest_metrics = results
                            
                            # Create a cumulative return series for charting
                            data['Cumulative_Return'] = 1 + (eval_results['roi'] / 100)
                            
                            # Store in session state
                            st.session_state.backtest_results = data
                            
                        except Exception as e:
                            st.error(f"Error evaluating AI model: {str(e)}")
                            continue_execution = False
                    
                    # If there was an error, skip to displaying results
                    if not continue_execution:
                        st.warning("Couldn't complete the backtest due to errors")
                        st.stop()
                    else:
                        # Create signals
                        data['Signal'] = 0
                        if strategy_type == "Moving Average Crossover":
                            # Use the MA columns or calculate them if needed
                            if 'MA_20' not in data.columns or 'MA_50' not in data.columns:
                                data[f'MA_{params["fast_ma"]}'] = data['Close'].rolling(window=params["fast_ma"]).mean()
                                data[f'MA_{params["slow_ma"]}'] = data['Close'].rolling(window=params["slow_ma"]).mean()
                                fast_col = f'MA_{params["fast_ma"]}'
                                slow_col = f'MA_{params["slow_ma"]}'
                            else:
                                fast_col = 'MA_20'
                                slow_col = 'MA_50'
                            
                            data.loc[data[fast_col] > data[slow_col], 'Signal'] = 1  # Buy signal
                            data.loc[data[fast_col] < data[slow_col], 'Signal'] = -1  # Sell signal
                        
                        elif strategy_type == "RSI Mean Reversion":
                            # Use RSI column or calculate it if needed
                            if 'RSI' not in data.columns:
                                # Calculate RSI with the specified period
                                delta = data['Close'].diff()
                                gain = delta.where(delta > 0, 0)
                                loss = -delta.where(delta < 0, 0)
                                avg_gain = gain.rolling(window=params["rsi_period"]).mean()
                                avg_loss = loss.rolling(window=params["rsi_period"]).mean()
                                rs = avg_gain / avg_loss
                                data['RSI'] = 100 - (100 / (1 + rs))
                            
                            data.loc[data['RSI'] < params["rsi_oversold"], 'Signal'] = 1  # Buy when oversold
                            data.loc[data['RSI'] > params["rsi_overbought"], 'Signal'] = -1  # Sell when overbought
                        
                        else:  # Combined Strategy
                            # Combined - simple version
                            ma_signal = np.zeros(len(data))
                            rsi_signal = np.zeros(len(data))
                            
                            if params["use_ma"]:
                                # Calculate MAs if needed
                                if f'MA_{params["fast_ma"]}' not in data.columns or f'MA_{params["slow_ma"]}' not in data.columns:
                                    data[f'MA_{params["fast_ma"]}'] = data['Close'].rolling(window=params["fast_ma"]).mean()
                                    data[f'MA_{params["slow_ma"]}'] = data['Close'].rolling(window=params["slow_ma"]).mean()
                                
                                ma_signal[(data[f'MA_{params["fast_ma"]}'] > data[f'MA_{params["slow_ma"]}'])] = 1
                                ma_signal[(data[f'MA_{params["fast_ma"]}'] < data[f'MA_{params["slow_ma"]}'])] = -1
                            
                            if params["use_rsi"]:
                                # Calculate RSI if needed
                                if 'RSI' not in data.columns:
                                    delta = data['Close'].diff()
                                    gain = delta.where(delta > 0, 0)
                                    loss = -delta.where(delta < 0, 0)
                                    avg_gain = gain.rolling(window=params["rsi_period"]).mean()
                                    avg_loss = loss.rolling(window=params["rsi_period"]).mean()
                                    rs = avg_gain / avg_loss
                                    data['RSI'] = 100 - (100 / (1 + rs))
                                
                                rsi_signal[data['RSI'] < params["rsi_oversold"]] = 1
                                rsi_signal[data['RSI'] > params["rsi_overbought"]] = -1
                            
                            if params["use_ma"] and params["use_rsi"]:
                                # Combined signal
                                data['Signal'] = np.where(ma_signal == rsi_signal, ma_signal, 0)
                            elif params["use_ma"]:
                                data['Signal'] = ma_signal
                            elif params["use_rsi"]:
                                data['Signal'] = rsi_signal
                        
                        # Calculate strategy returns
                        if 'Returns' not in data.columns:
                            data['Returns'] = data['Close'].pct_change()
                        
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
                                duration = (exit_date - entry_date).days if isinstance(exit_date, pd.Timestamp) and isinstance(entry_date, pd.Timestamp) else 0
                                
                                trade = {
                                    'entry_date': entry_date,
                                    'exit_date': exit_date,
                                    'entry_price': entry_price,
                                    'exit_price': exit_price,
                                    'direction': position,
                                    'profit_pct': profit_pct,
                                    'duration': duration
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
                    
                    # Add regime integration
                    with st.expander("Market Regime Integration"):
                        st.subheader("Strategy Performance by Market Regime")
                        
                        # Detect market regimes for the backtest period
                        try:
                            detector = market_regime.MarketRegimeDetector()
                            
                            # Run regime detection on backtest data
                            regimes = []
                            lookback = 20  # Lookback window for regime detection
                            
                            # Use a sliding window approach to detect regimes throughout the backtest period
                            for i in range(lookback, len(data), lookback // 2):  # Overlap windows by half
                                window_data = data.iloc[i-lookback:i].copy()
                                if len(window_data) >= lookback // 2:  # Ensure sufficient data
                                    regime = detector.detect_regime(window_data, lookback_period=len(window_data))
                                    regimes.append({
                                        'date': data.index[i-1],
                                        'regime': regime['regime'],
                                        'confidence': regime['confidence']
                                    })
                            
                            if regimes:
                                # Create a DataFrame of detected regimes
                                regime_df = pd.DataFrame(regimes)
                                
                                st.write("Detected Market Regimes During Backtest:")
                                st.dataframe(regime_df)
                                
                                # Option to update strategy performance for detected regimes
                                if st.checkbox("Update Strategy Performance for Detected Regimes", False):
                                    st.write("This will save the strategy's performance metrics for each detected market regime")
                                    
                                    if st.button("Save Performance Data by Regime"):
                                        # Group the backtest data by regime and calculate performance for each
                                        regime_performances = {}
                                        
                                        for reg in regimes:
                                            regime_type = reg['regime']
                                            date = reg['date']
                                            
                                            # Find data for this regime period (between this regime point and the next)
                                            next_idx = regimes.index(reg) + 1
                                            end_date = regimes[next_idx]['date'] if next_idx < len(regimes) else data.index[-1]
                                            
                                            try:
                                                # Get data for this regime period
                                                period_data = data.loc[date:end_date].copy()
                                                
                                                if len(period_data) > 5:  # Ensure sufficient data
                                                    # Calculate performance for this period
                                                    period_return = period_data['Cumulative_Return'].iloc[-1] / period_data['Cumulative_Return'].iloc[0] - 1
                                                    
                                                    # Calculate other metrics
                                                    period_returns = period_data['Strategy_Return'].dropna()
                                                    period_sharpe = period_returns.mean() / period_returns.std() * np.sqrt(252) if period_returns.std() > 0 else 0
                                                    
                                                    # Store metrics
                                                    if regime_type not in regime_performances:
                                                        regime_performances[regime_type] = {
                                                            'returns': [],
                                                            'sharpe_ratios': [],
                                                            'periods': 0
                                                        }
                                                    
                                                    regime_performances[regime_type]['returns'].append(period_return)
                                                    regime_performances[regime_type]['sharpe_ratios'].append(period_sharpe)
                                                    regime_performances[regime_type]['periods'] += 1
                                            except Exception as e:
                                                st.warning(f"Could not calculate performance for period {date} to {end_date}: {str(e)}")
                                        
                                        # Calculate average performance by regime
                                        for regime_type, perf in regime_performances.items():
                                            if perf['periods'] > 0:
                                                avg_return = np.mean(perf['returns'])
                                                avg_sharpe = np.mean(perf['sharpe_ratios'])
                                                
                                                # Create performance metrics
                                                metrics = {
                                                    'total_return': avg_return * 100,
                                                    'sharpe_ratio': avg_sharpe,
                                                    'sample_size': perf['periods']
                                                }
                                                
                                                # Update strategy performance in detector
                                                detector.update_strategy_performance(
                                                    regime_type, 
                                                    {
                                                        'strategy_type': strategy_type,
                                                        'parameters': params
                                                    }, 
                                                    metrics
                                                )
                                        
                                        st.success("Strategy performance saved for each detected regime!")
                                        
                                # Option to get optimal strategy for a specific regime
                                st.subheader("Optimal Strategy by Regime")
                                regime_options = list(set([r['regime'] for r in regimes]))
                                if regime_options:
                                    selected_regime = st.selectbox("Select a market regime:", regime_options)
                                    
                                    if st.button(f"Get Optimal Strategy for {selected_regime}"):
                                        try:
                                            optimal_params = detector.get_best_strategy(selected_regime)
                                            
                                            if optimal_params:
                                                st.success(f"Found optimized strategy for {selected_regime} regime!")
                                                
                                                # Display strategy parameters
                                                st.write("**Optimal Strategy Parameters:**")
                                                
                                                if "strategy_type" in optimal_params:
                                                    st.write(f"Strategy Type: {optimal_params['strategy_type']}")
                                                    
                                                    if "parameters" in optimal_params:
                                                        for param, value in optimal_params["parameters"].items():
                                                            st.write(f"- {param}: {value}")
                                                else:
                                                    # If it's just parameters without type
                                                    for param, value in optimal_params.items():
                                                        st.write(f"- {param}: {value}")
                                            else:
                                                st.info(f"No optimized strategy available for {selected_regime} regime yet.")
                                        except Exception as e:
                                            st.error(f"Error retrieving optimal strategy: {str(e)}")
                            else:
                                st.info("No regimes detected during the backtest period.")
                        except Exception as e:
                            st.error(f"Error in regime detection: {str(e)}")
        
        with save_col:
            # Only enable save button if backtest has been run
            if 'backtest_metrics' in st.session_state:
                if st.button("Save Strategy & Results to Storage"):
                    try:
                        # Save strategy to storage
                        strategy_id = storage.save_strategy(
                            strategy_name,
                            strategy_type,
                            strategy_description,
                            params
                        )
                        
                        # Save backtest results
                        start_datetime = datetime.combine(start_date, datetime.min.time())
                        end_datetime = datetime.combine(end_date, datetime.max.time())
                        
                        backtest_id = storage.save_backtest_result(
                            strategy_id,
                            symbol,
                            timeframe,
                            start_datetime,
                            end_datetime,
                            st.session_state.backtest_metrics
                        )
                        
                        st.success(f"Strategy and results saved to storage (ID: {strategy_id})")
                    except Exception as e:
                        st.error(f"Error saving to storage: {str(e)}")
    
    # Training Tab (New)
    elif st.session_state.current_tab == 'Training':
        st.header("AI Model Training")
        
        if st.session_state.data_loaded:
            # Training parameters
            st.subheader("Training Parameters")
            
            episodes = st.slider("Training Episodes", 1, 50, 10)
            batch_size = st.slider("Batch Size", 8, 128, 32)
            
            # Evolution parameters
            save_evolution = st.checkbox("Save Evolution Checkpoints", True, 
                                        help="Save intermediate models during training to visualize learning progression")
            
            if save_evolution:
                evolution_interval = st.slider("Evolution Checkpoint Interval", 1, 10, 2, 
                                             help="Number of episodes between evolution checkpoints")
            else:
                evolution_interval = 0
            
            # Train-test split
            split_ratio = st.slider("Training Data Percentage", 50, 90, 80) / 100
            
            # Training section
            if st.button("Train AI Model"):
                with st.spinner("Training AI model..."):
                    try:
                        # Prepare data
                        data = st.session_state.market_data.copy()
                        
                        # Drop NaN values
                        data = data.dropna()
                        
                        # Split data into train and test
                        split_idx = int(len(data) * split_ratio)
                        train_data = data.iloc[:split_idx]
                        test_data = data.iloc[split_idx:]
                        
                        # Train the model
                        st.text("Training in progress...")
                        model_name = f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        agent, training_results = simple_rl.train_agent(
                            train_data, 
                            episodes=episodes, 
                            batch_size=batch_size, 
                            save_model_name=model_name,
                            save_evolution=save_evolution,
                            evolution_interval=evolution_interval
                        )
                        
                        # Store training results in session state for Strategy Evolution tab
                        st.session_state.training_results = training_results
                        st.session_state.last_trained_model = model_name
                        
                        # Evaluate on test data
                        st.text("Evaluating model on test data...")
                        eval_results = simple_rl.evaluate_agent(agent, test_data)
                        
                        # Display results
                        st.success(f"Model trained and saved as {model_name}")
                        
                        # Update session state
                        st.session_state.model_trained = True
                        st.session_state.current_model = f"{model_name}.json"
                        
                        # Display training metrics
                        st.subheader("Training Results")
                        
                        # Create metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Test ROI", f"{eval_results['roi']:.2f}%")
                        with col2:
                            st.metric("Number of Trades", eval_results['trades'])
                        with col3:
                            if eval_results['roi'] > eval_results['hold_roi']:
                                outperformance = eval_results['roi'] - eval_results['hold_roi']
                                st.metric("vs. Buy & Hold", f"+{outperformance:.2f}%")
                            else:
                                underperformance = eval_results['hold_roi'] - eval_results['roi']
                                st.metric("vs. Buy & Hold", f"-{underperformance:.2f}%")
                                
                        # Display training progress
                        st.subheader("Training Progress")
                        
                        # Create a DataFrame for training metrics
                        progress_df = pd.DataFrame({
                            'Episode': range(1, episodes + 1),
                            'Reward': training_results['episode_rewards'],
                            'Portfolio Value': training_results['portfolio_values'],
                            'Trades': training_results['trade_counts']
                        })
                        
                        # Plot training progress
                        st.line_chart(progress_df.set_index('Episode')[['Reward']])
                        st.line_chart(progress_df.set_index('Episode')[['Portfolio Value']])
                        
                        if save_evolution and training_results['evolution_models']:
                            st.success(f"Saved {len(training_results['evolution_models'])} evolution checkpoints. Go to the Strategy Evolution tab to explore them.")
                        
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
            
            # Model management section
            st.subheader("Model Management")
            
            # Get available models
            available_models = simple_rl.get_all_models()
            
            if not available_models:
                st.info("No trained models found")
            else:
                # Display models
                st.write(f"Found {len(available_models)} trained models")
                
                # Model selection
                selected_model = st.selectbox("Select Model", available_models)
                
                # Load selected model
                if st.button("Load Selected Model"):
                    try:
                        # Load model and update session state
                        model_path = os.path.join(simple_rl.MODELS_DIR, selected_model)
                        agent = simple_rl.TradingRLAgent.load_model(model_path)
                        
                        st.session_state.model_trained = True
                        st.session_state.current_model = selected_model
                        
                        st.success(f"Model {selected_model} loaded successfully")
                    except Exception as e:
                        st.error(f"Error loading model: {str(e)}")
        else:
            st.warning("Please load market data first")
            
    # Strategy Evolution Tab
    elif st.session_state.current_tab == 'Strategy Evolution':
        st.header("Strategy Evolution")
        
        # Check if we have evolution data
        if 'training_results' in st.session_state and 'evolution_models' in st.session_state.training_results and st.session_state.training_results['evolution_models']:
            # Get evolution data
            evolution_models = st.session_state.training_results['evolution_models']
            
            st.subheader("Learning Progress Over Time")
            
            # Create a DataFrame for evolution metrics
            evolution_df = pd.DataFrame([
                {
                    'Episode': e['episode'],
                    'Portfolio Value': e['portfolio_value'],
                    'Reward': e['reward'],
                    'Win Rate': e.get('win_rate', 0) * 100 if isinstance(e.get('win_rate', 0), float) else e.get('win_rate', 0),
                    'Trades': e['trades'],
                    'Avg Profit': e.get('avg_profit', 0),
                    'Exploration Rate': e['epsilon'],
                    'Model Path': e['model_path']
                } for e in evolution_models
            ])
            
            # Show evolution metrics as a table
            st.dataframe(evolution_df[['Episode', 'Portfolio Value', 'Reward', 'Win Rate', 'Trades', 'Exploration Rate']], 
                        use_container_width=True)
            
            # Create line charts to show progression of key metrics
            st.subheader("Learning Curves")
            
            # Create tabs for different metrics
            tab1, tab2, tab3, tab4 = st.tabs(["Portfolio Value", "Reward", "Win Rate", "Exploration Rate"])
            
            with tab1:
                st.line_chart(evolution_df.set_index('Episode')[['Portfolio Value']])
                st.write("As the model learns, it should generate higher portfolio values over time.")
                
            with tab2:
                st.line_chart(evolution_df.set_index('Episode')[['Reward']])
                st.write("Reward represents the cumulative rewards earned during each episode. Higher is better.")
                
            with tab3:
                st.line_chart(evolution_df.set_index('Episode')[['Win Rate']])
                st.write("Win rate shows the percentage of profitable trades. Should increase as the model improves.")
                
            with tab4:
                st.line_chart(evolution_df.set_index('Episode')[['Exploration Rate']])
                st.write("Exploration rate (epsilon) decreases over time as the model shifts from exploration to exploitation.")
            
            # Evolution model analysis
            st.subheader("Evolution Model Analysis")
            
            # Select two models to compare
            st.write("Compare models from different stages of evolution:")
            col1, col2 = st.columns(2)
            
            with col1:
                early_model_idx = st.selectbox("Select Early Model", 
                                             range(len(evolution_models)),
                                             format_func=lambda x: f"Episode {evolution_models[x]['episode']}")
                early_model = evolution_models[early_model_idx]
                
            with col2:
                late_model_idx = st.selectbox("Select Later Model", 
                                            range(len(evolution_models)),
                                            index=len(evolution_models)-1,
                                            format_func=lambda x: f"Episode {evolution_models[x]['episode']}")
                late_model = evolution_models[late_model_idx]
            
            # Show comparison
            if early_model_idx != late_model_idx:
                st.subheader("Model Comparison")
                
                # Calculate differences
                portfolio_change = ((late_model['portfolio_value'] - early_model['portfolio_value']) / 
                                    early_model['portfolio_value'] * 100)
                reward_change = late_model['reward'] - early_model['reward']
                win_rate_change = (late_model.get('win_rate', 0) - early_model.get('win_rate', 0)) * 100 if isinstance(late_model.get('win_rate', 0), float) else 0
                
                # Display comparisons
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Portfolio Value Change", 
                             f"{portfolio_change:.2f}%", 
                             delta=f"{portfolio_change:.2f}%")
                with col2:
                    st.metric("Reward Change", 
                             f"{reward_change:.2f}", 
                             delta=f"{reward_change:.2f}")
                with col3:
                    st.metric("Win Rate Change", 
                             f"{win_rate_change:.2f}%", 
                             delta=f"{win_rate_change:.2f}%")
                
                # Option to backtest these models
                if st.button("Evaluate Selected Models"):
                    with st.spinner("Evaluating models..."):
                        try:
                            # Load both models
                            early_agent = simple_rl.TradingRLAgent.load_model(early_model['model_path'])
                            late_agent = simple_rl.TradingRLAgent.load_model(late_model['model_path'])
                            
                            # Get test data (use the last 20% of the data)
                            if st.session_state.data_loaded:
                                data = st.session_state.market_data.copy().dropna()
                                split_idx = int(len(data) * 0.8)
                                test_data = data.iloc[split_idx:]
                                
                                # Evaluate both models
                                early_results = simple_rl.evaluate_agent(early_agent, test_data)
                                late_results = simple_rl.evaluate_agent(late_agent, test_data)
                                
                                # Display results
                                st.subheader("Backtest Results")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"### Early Model (Episode {early_model['episode']})")
                                    st.metric("ROI", f"{early_results['roi']:.2f}%")
                                    st.metric("Trades", early_results['trades'])
                                    
                                with col2:
                                    st.write(f"### Later Model (Episode {late_model['episode']})")
                                    st.metric("ROI", f"{late_results['roi']:.2f}%")
                                    st.metric("Trades", late_results['trades'])
                                
                                # Calculate improvement
                                roi_improvement = late_results['roi'] - early_results['roi']
                                
                                st.info(f"The later model {late_model['episode']} {'outperforms' if roi_improvement > 0 else 'underperforms'} the early model {early_model['episode']} by {abs(roi_improvement):.2f}% ROI.")
                            else:
                                st.warning("Please load market data first to evaluate models")
                                
                        except Exception as e:
                            st.error(f"Error evaluating models: {str(e)}")
            
            # Strategy insights
            st.subheader("Strategy Insights")
            
            st.write("""
            ## How Models Evolve
            
            1. **Exploration to Exploitation**: The model starts by exploring random actions to discover profitable strategies, 
               then gradually shifts to exploiting known profitable patterns.
            
            2. **Learning from Mistakes**: Early models make many mistakes but learn from them to improve over time.
            
            3. **Pattern Recognition**: As training progresses, the model learns to recognize price patterns and indicators 
               that have historically led to profitable trades.
            
            4. **Risk Management**: Later models typically develop better risk management, with higher win rates and 
               more consistent returns.
            """)
            
        elif 'training_results' in st.session_state:
            st.warning("No evolution data found. Train a model with 'Save Evolution Checkpoints' enabled.")
        else:
            st.info("No training has been performed yet. Go to the Training tab to train a model with evolution tracking.")
            
            # Show example of what to expect
            st.subheader("Example Evolution")
            
            # Create sample data for illustration
            sample_episodes = range(1, 11)
            sample_portfolio = [10000 * (1 + i * 0.05 + np.random.normal(0, 0.02)) for i in range(10)]
            sample_win_rates = [40 + i * 5 + np.random.normal(0, 2) for i in range(10)]
            
            # Create sample DataFrame
            sample_df = pd.DataFrame({
                'Episode': sample_episodes,
                'Portfolio Value': sample_portfolio,
                'Win Rate (%)': sample_win_rates
            })
            
            # Show sample charts
            st.line_chart(sample_df.set_index('Episode')[['Portfolio Value']])
            st.line_chart(sample_df.set_index('Episode')[['Win Rate (%)']])
            
            st.write("""
            This is an example of how the strategy evolution charts will look after training.
            When you train a model with 'Save Evolution Checkpoints' enabled, you'll see the actual
            learning progression of your model here.
            """)
    
    # Live Trading Tab
    elif st.session_state.current_tab == 'Live Trading':
        st.header("Live Trading")
        
        if not st.session_state.model_trained or not st.session_state.current_model:
            st.warning("Please train or load an AI model first")
        else:
            st.success(f"Model ready: {st.session_state.current_model}")
            
            # Trading parameters
            st.subheader("Trading Parameters")
            
            initial_capital = st.number_input("Initial Capital", min_value=100, max_value=1000000, value=10000)
            max_position_size = st.slider("Maximum Position Size (%)", 1, 100, 20) / 100
            
            # API connection
            st.subheader("Broker Connection")
            
            # Placeholder for API keys (in a real app, these would be securely handled)
            api_key = st.text_input("API Key", type="password")
            api_secret = st.text_input("API Secret", type="password")
            
            api_status = st.empty()
            
            if st.button("Connect to Broker"):
                if api_key and api_secret:
                    # Placeholder for API connection
                    api_status.success("✅ Connected to broker API (simulation)")
                else:
                    api_status.error("❌ API credentials required")
            
            # Trading control
            st.subheader("Trading Control")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Start Automated Trading"):
                    if api_key and api_secret:
                        st.success("🤖 Trading bot started (simulation)")
                        st.info("The bot would now trade autonomously based on the trained model")
                    else:
                        st.error("❌ Please connect to broker first")
            
            with col2:
                if st.button("Stop Trading"):
                    st.warning("⏹ Trading bot stopped (simulation)")
            
            # Trading status
            st.subheader("Trading Status")
            
            # Simulated trading status
            status_col1, status_col2, status_col3 = st.columns(3)
            
            with status_col1:
                st.metric("Account Value", f"${initial_capital:.2f}")
            with status_col2:
                st.metric("Open Positions", "0")
            with status_col3:
                st.metric("Today's P&L", "+$0.00")
            
            # Simulated recent trades
            st.subheader("Recent Trades (Simulation)")
            
            # Create empty dataframe for trades
            trades_df = pd.DataFrame({
                'Date': [],
                'Symbol': [],
                'Type': [],
                'Price': [],
                'Size': [],
                'P&L': []
            })
            
            st.dataframe(trades_df, use_container_width=True)
            
            st.info("When connected to a real broker, the bot would execute trades based on the AI model's decisions")
    
    # Performance Tab
    elif st.session_state.current_tab == 'Performance':
        st.header("Performance Analytics")
        
        if 'backtest_results' not in st.session_state or st.session_state.backtest_results is None:
            # Try to load from storage
            try:
                backtest_results = storage.get_backtest_results()
                
                if not backtest_results:
                    st.info("No backtest results found in storage. Run a backtest first.")
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
                    if 'strategy_name' in display_df.columns:
                        st.dataframe(
                            display_df[['strategy_name', 'symbol', 'timeframe', 'total_return', 
                                       'annualized_return', 'sharpe_ratio', 'max_drawdown', 
                                       'win_rate', 'trades_count', 'execution_date']], 
                            use_container_width=True
                        )
                    else:
                        st.dataframe(display_df, use_container_width=True)
                    
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
                        
                        # Display trades if available
                        if 'trades' in selected_result:
                            st.subheader("Trades")
                            trades = selected_result['trades']
                            if trades:
                                trades_df = pd.DataFrame(trades)
                                st.dataframe(trades_df, use_container_width=True)
                        
                        # Note about equity curve not being available for storage results
                        st.info("Detailed equity curve data is not stored. Run a new backtest to see the equity curve.")
            except Exception as e:
                st.error(f"Error loading backtest results from storage: {str(e)}")
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
            monthly_returns = data['Strategy_Return'].resample('ME').apply(lambda x: (1 + x).prod() - 1)
            
            # Display as a bar chart
            st.bar_chart(monthly_returns)
    
    # Storage Tab (renamed from Database)
    elif st.session_state.current_tab == 'Storage':
        st.header("Data Storage Management")
        
        tab1, tab2, tab3 = st.tabs(["Strategies", "Market Data", "AI Models"])
        
        with tab1:
            st.subheader("Saved Strategies")
            
            # Refresh button
            if st.button("Refresh Strategies"):
                st.rerun()
            
            # List all strategies
            try:
                strategies = storage.get_strategies()
                
                if not strategies:
                    st.info("No strategies found in storage.")
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
                            backtest_results = storage.get_backtest_results(selected_strategy_id)
                            
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
                                if 'symbol' in results_df.columns:
                                    st.dataframe(
                                        results_df[['symbol', 'timeframe', 'total_return', 'annualized_return', 
                                                  'sharpe_ratio', 'win_rate', 'trades_count', 'execution_date']], 
                                        use_container_width=True
                                    )
                                else:
                                    st.dataframe(results_df, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error loading backtest results: {str(e)}")
            except Exception as e:
                st.error(f"Error loading strategies from storage: {str(e)}")
        
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
            
            if st.button("Check Storage"):
                try:
                    # Check if the file exists
                    filename = f"{db_symbol}_{db_timeframe}.csv"
                    filepath = os.path.join(storage.MARKET_DATA_DIR, filename)
                    
                    if os.path.exists(filepath):
                        df = pd.read_csv(filepath)
                        st.success(f"Found {len(df)} records for {db_symbol} ({db_timeframe}) in storage")
                    else:
                        st.warning(f"No data found for {db_symbol} ({db_timeframe}) in storage")
                except Exception as e:
                    st.error(f"Error checking storage: {str(e)}")
        
        with tab3:
            st.subheader("AI Models Management")
            
            # Initialize models directory
            os.makedirs(simple_rl.MODELS_DIR, exist_ok=True)
            
            # Get available models
            available_models = simple_rl.get_all_models()
            
            if not available_models:
                st.info("No trained AI models found")
            else:
                # Display models
                st.write(f"Found {len(available_models)} trained models")
                
                # Create a dataframe with model details
                models_info = []
                
                for model_name in available_models:
                    try:
                        # Load model details
                        model_path = os.path.join(simple_rl.MODELS_DIR, model_name)
                        with open(model_path, 'r') as f:
                            model_data = json.load(f)
                            
                        # Extract model info
                        model_info = {
                            'name': model_name,
                            'state_size': model_data.get('state_size', 'N/A'),
                            'action_size': model_data.get('action_size', 'N/A'),
                            'learning_rate': model_data.get('learning_rate', 'N/A'),
                            'saved_date': model_data.get('saved_date', 'N/A')
                        }
                        
                        models_info.append(model_info)
                    except Exception as e:
                        st.error(f"Error loading model {model_name}: {str(e)}")
                
                # Display models table
                if models_info:
                    models_df = pd.DataFrame(models_info)
                    st.dataframe(models_df, use_container_width=True)
                    
                    # Model selection
                    selected_model = st.selectbox("Select Model", available_models, key="model_mgmt")
                    
                    # Model actions
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Load Selected Model"):
                            try:
                                # Load model and update session state
                                model_path = os.path.join(simple_rl.MODELS_DIR, selected_model)
                                agent = simple_rl.TradingRLAgent.load_model(model_path)
                                
                                st.session_state.model_trained = True
                                st.session_state.current_model = selected_model
                                
                                st.success(f"Model {selected_model} loaded successfully")
                            except Exception as e:
                                st.error(f"Error loading model: {str(e)}")
                    
                    with col2:
                        if st.button("Delete Selected Model"):
                            try:
                                # Delete model file
                                model_path = os.path.join(simple_rl.MODELS_DIR, selected_model)
                                os.remove(model_path)
                                
                                # Update current model if it was deleted
                                if st.session_state.current_model == selected_model:
                                    st.session_state.model_trained = False
                                    st.session_state.current_model = None
                                
                                st.success(f"Model {selected_model} deleted")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting model: {str(e)}")
            
            # Storage maintenance section
            st.subheader("Storage Maintenance")
            
            maintenance_option = st.radio(
                "Maintenance Options",
                ["Information", "Initialize Storage"]
            )
            
            if maintenance_option == "Information":
                st.info("""
                The file-based storage system stores:
                - Market data for different symbols and timeframes (CSV files)
                - Strategy configurations (JSON files)
                - Backtest results and trades (JSON files)
                - AI models trained with reinforcement learning (JSON files)
                
                Data is stored in separate directories for each type.
                """)
            elif maintenance_option == "Initialize Storage":
                if st.button("Initialize Storage System"):
                    try:
                        storage.initialize_storage()
                        os.makedirs(simple_rl.MODELS_DIR, exist_ok=True)
                        st.success("Storage system initialized successfully")
                    except Exception as e:
                        st.error(f"Error initializing storage: {str(e)}")
                        
            st.markdown("""
            ---
            **Note:** This is a simple file-based storage system. For production use,
            a more robust database system would be recommended.
            """)