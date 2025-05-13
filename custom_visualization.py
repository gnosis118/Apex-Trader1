import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_candlestick_chart(data, ma_fast=20, ma_slow=50, bb_period=20, bb_std=2.0, buy_signals=None, sell_signals=None):
    """
    Create a candlestick chart with technical indicators
    
    Parameters:
    -----------
    data: pandas.DataFrame
        DataFrame containing OHLCV data and indicators
    ma_fast: int
        Fast moving average period
    ma_slow: int
        Slow moving average period
    bb_period: int
        Bollinger Bands period
    bb_std: float
        Bollinger Bands standard deviation
    buy_signals: list
        List of buy signal timestamps
    sell_signals: list
        List of sell signal timestamps
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The candlestick chart figure
    """
    # Create figure
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        )
    )
    
    # Add moving averages if they exist in the data
    ma_fast_col = f'MA_{ma_fast}'
    ma_slow_col = f'MA_{ma_slow}'
    
    if ma_fast_col in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[ma_fast_col],
                name=f'{ma_fast}-period MA',
                line=dict(color='rgba(255, 165, 0, 0.7)', width=1)
            )
        )
    
    if ma_slow_col in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[ma_slow_col],
                name=f'{ma_slow}-period MA',
                line=dict(color='rgba(73, 133, 231, 0.7)', width=1)
            )
        )
    
    # Add Bollinger Bands if they exist in the data
    bb_upper_col = f'BB_Upper_{bb_period}'
    bb_middle_col = f'BB_Middle_{bb_period}'
    bb_lower_col = f'BB_Lower_{bb_period}'
    
    if bb_upper_col in data.columns and bb_middle_col in data.columns and bb_lower_col in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[bb_upper_col],
                name='BB Upper',
                line=dict(color='rgba(173, 204, 255, 0.5)', width=1)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[bb_middle_col],
                name='BB Middle',
                line=dict(color='rgba(173, 204, 255, 0.5)', width=1)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[bb_lower_col],
                name='BB Lower',
                line=dict(color='rgba(173, 204, 255, 0.5)', width=1),
                fill='tonexty',
                fillcolor='rgba(173, 204, 255, 0.1)'
            )
        )
    
    # Add buy signals if provided
    if buy_signals:
        buy_x = [x for x in buy_signals if x in data.index]
        if buy_x:
            buy_y = [data.loc[x, 'Low'] * 0.99 for x in buy_x]  # Slightly below the candle
            
            fig.add_trace(
                go.Scatter(
                    x=buy_x,
                    y=buy_y,
                    mode='markers',
                    name='Buy',
                    marker=dict(
                        symbol='triangle-up',
                        size=10,
                        color='green',
                        line=dict(color='green', width=1)
                    )
                )
            )
    
    # Add sell signals if provided
    if sell_signals:
        sell_x = [x for x in sell_signals if x in data.index]
        if sell_x:
            sell_y = [data.loc[x, 'High'] * 1.01 for x in sell_x]  # Slightly above the candle
            
            fig.add_trace(
                go.Scatter(
                    x=sell_x,
                    y=sell_y,
                    mode='markers',
                    name='Sell',
                    marker=dict(
                        symbol='triangle-down',
                        size=10,
                        color='red',
                        line=dict(color='red', width=1)
                    )
                )
            )
    
    # Update layout
    fig.update_layout(
        title='Price Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Set y-axis to log scale for better visualization
    fig.update_yaxes(type='log')
    
    return fig

def create_technical_indicator_chart(data, indicator_type='momentum'):
    """
    Create a chart for a specific type of technical indicator
    
    Parameters:
    -----------
    data: pandas.DataFrame
        DataFrame containing technical indicators
    indicator_type: str
        Type of indicator to plot ('momentum', 'volatility', or 'trend')
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The indicator chart figure
    """
    # Create figure
    fig = go.Figure()
    
    if indicator_type == 'momentum':
        # RSI
        if 'RSI_14' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['RSI_14'],
                    name='RSI (14)',
                    line=dict(color='purple', width=1)
                )
            )
            
            # Add overbought/oversold lines
            fig.add_shape(
                type="line",
                x0=data.index[0],
                y0=70,
                x1=data.index[-1],
                y1=70,
                line=dict(color="red", width=1, dash="dash"),
            )
            
            fig.add_shape(
                type="line",
                x0=data.index[0],
                y0=30,
                x1=data.index[-1],
                y1=30,
                line=dict(color="green", width=1, dash="dash"),
            )
        
        # MACD
        if all(col in data.columns for col in ['MACD', 'MACD_Signal', 'MACD_Hist']):
            # Create a secondary y-axis figure for MACD
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add RSI
            if 'RSI_14' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['RSI_14'],
                        name='RSI (14)',
                        line=dict(color='purple', width=1)
                    ),
                    secondary_y=False
                )
                
                # Add overbought/oversold lines
                fig.add_shape(
                    type="line",
                    x0=data.index[0],
                    y0=70,
                    x1=data.index[-1],
                    y1=70,
                    line=dict(color="red", width=1, dash="dash"),
                )
                
                fig.add_shape(
                    type="line",
                    x0=data.index[0],
                    y0=30,
                    x1=data.index[-1],
                    y1=30,
                    line=dict(color="green", width=1, dash="dash"),
                )
            
            # Add MACD line
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACD'],
                    name='MACD',
                    line=dict(color='blue', width=1)
                ),
                secondary_y=True
            )
            
            # Add MACD signal line
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACD_Signal'],
                    name='Signal',
                    line=dict(color='red', width=1)
                ),
                secondary_y=True
            )
            
            # Add MACD histogram
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['MACD_Hist'],
                    name='Histogram',
                    marker=dict(
                        color=np.where(data['MACD_Hist'] >= 0, 'green', 'red'),
                        opacity=0.5
                    )
                ),
                secondary_y=True
            )
            
            # Update layout for RSI
            fig.update_yaxes(
                title_text="RSI",
                range=[0, 100],
                secondary_y=False
            )
            
            # Update layout for MACD
            fig.update_yaxes(
                title_text="MACD",
                secondary_y=True
            )
    
    elif indicator_type == 'volatility':
        # Bollinger Bands Width
        if 'BB_Width_20' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Width_20'],
                    name='BB Width (20)',
                    line=dict(color='blue', width=1)
                )
            )
        
        # ATR
        if 'ATR_14' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['ATR_14'],
                    name='ATR (14)',
                    line=dict(color='orange', width=1)
                )
            )
        
        # Historical Volatility (if exists)
        if 'Volatility' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Volatility'] * 100,  # Convert to percentage
                    name='Historical Volatility (20)',
                    line=dict(color='purple', width=1)
                )
            )
    
    elif indicator_type == 'trend':
        # Moving Averages
        for period in [20, 50, 100, 200]:
            col = f'MA_{period}'
            if col in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[col],
                        name=f'MA ({period})',
                        line=dict(width=1)
                    )
                )
    
    # Update layout
    if indicator_type == 'momentum' and not ('MACD' in data.columns and 'RSI_14' in data.columns):
        title = 'Momentum Indicators - RSI'
        y_title = 'RSI'
        y_range = [0, 100]
    elif indicator_type == 'momentum':
        title = 'Momentum Indicators - RSI and MACD'
        y_title = None  # Already set in the secondary y-axis
        y_range = None
    elif indicator_type == 'volatility':
        title = 'Volatility Indicators'
        y_title = 'Value'
        y_range = None
    elif indicator_type == 'trend':
        title = 'Trend Indicators'
        y_title = 'Value'
        y_range = None
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=y_title,
        template='plotly_dark',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    if y_range:
        fig.update_yaxes(range=y_range)
    
    return fig

def create_performance_chart(equity_curve, drawdowns=None, chart_type='equity'):
    """
    Create a performance chart (equity curve, drawdowns, etc.)
    
    Parameters:
    -----------
    equity_curve: pandas.Series
        Series containing equity curve values
    drawdowns: pandas.Series
        Series containing drawdown values
    chart_type: str
        Type of chart ('equity', 'drawdown', 'equity_drawdown')
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The performance chart figure
    """
    # Create figure
    if chart_type == 'equity_drawdown':
        # Create plot with two y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()
    
    # Add equity curve
    if chart_type in ['equity', 'equity_drawdown']:
        if isinstance(equity_curve, pd.Series):
            x = equity_curve.index
            y = equity_curve.values
        else:
            x = list(range(len(equity_curve)))
            y = equity_curve
        
        if chart_type == 'equity_drawdown':
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name='Equity Curve',
                    line=dict(color='green', width=1)
                ),
                secondary_y=False
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name='Equity Curve',
                    line=dict(color='green', width=1)
                )
            )
    
    # Add drawdowns
    if chart_type in ['drawdown', 'equity_drawdown'] and drawdowns is not None:
        if isinstance(drawdowns, pd.Series):
            x = drawdowns.index
            y = drawdowns.values * 100  # Convert to percentage
        else:
            x = list(range(len(drawdowns)))
            y = drawdowns * 100  # Convert to percentage
        
        if chart_type == 'equity_drawdown':
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name='Drawdown',
                    line=dict(color='red', width=1),
                    fill='tozeroy',
                    fillcolor='rgba(255, 0, 0, 0.1)'
                ),
                secondary_y=True
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name='Drawdown',
                    line=dict(color='red', width=1),
                    fill='tozeroy',
                    fillcolor='rgba(255, 0, 0, 0.1)'
                )
            )
    
    # Update layout
    if chart_type == 'equity':
        title = 'Equity Curve'
        y_title = 'Portfolio Value'
    elif chart_type == 'drawdown':
        title = 'Drawdowns'
        y_title = 'Drawdown (%)'
    elif chart_type == 'equity_drawdown':
        title = 'Equity Curve and Drawdowns'
        
        # Set y-axis titles
        fig.update_yaxes(title_text="Portfolio Value", secondary_y=False)
        fig.update_yaxes(title_text="Drawdown (%)", secondary_y=True)
        
        # No need to set y_title separately
        y_title = None
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=y_title,
        template='plotly_dark',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_returns_heatmap(returns_series):
    """
    Create a heatmap of monthly returns
    
    Parameters:
    -----------
    returns_series: pandas.Series
        Series containing daily returns
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The heatmap figure
    """
    # Make sure the index is a datetime index
    if not isinstance(returns_series.index, pd.DatetimeIndex):
        raise ValueError("Returns series must have a DatetimeIndex")
    
    # Resample to monthly returns
    monthly_returns = returns_series.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # Create a pivot table (years x months)
    years = monthly_returns.index.year.unique()
    months = range(1, 13)
    
    # Create a 2D array of returns
    heatmap_data = np.zeros((len(years), 12))
    
    # Fill the array with return values
    for i, year in enumerate(years):
        for j, month in enumerate(months):
            # Find the return for this year and month
            monthly_idx = monthly_returns.index[
                (monthly_returns.index.year == year) & 
                (monthly_returns.index.month == month)
            ]
            
            if len(monthly_idx) > 0:
                heatmap_data[i, j] = monthly_returns.loc[monthly_idx[0]] * 100  # Convert to percentage
    
    # Create a heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        y=years,
        colorscale=[
            [0, 'rgb(255, 0, 0)'],     # Red for negative returns
            [0.5, 'rgb(255, 255, 255)'],  # White for zero
            [1, 'rgb(0, 255, 0)']      # Green for positive returns
        ],
        zmid=0,  # Center the color scale at zero
        text=[[f"{val:.2f}%" for val in row] for row in heatmap_data],
        hoverinfo='text'
    ))
    
    # Update layout
    fig.update_layout(
        title='Monthly Returns Heatmap',
        template='plotly_dark',
        xaxis=dict(
            title='Month',
            tickangle=-90
        ),
        yaxis=dict(
            title='Year',
            autorange='reversed'  # Most recent year at the top
        )
    )
    
    return fig

def create_trade_distribution_chart(trades):
    """
    Create a chart showing the distribution of trade returns
    
    Parameters:
    -----------
    trades: list
        List of trade dictionaries with 'profit_pct' key
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The distribution chart figure
    """
    if not trades:
        # Return empty figure if no trades
        fig = go.Figure()
        fig.update_layout(
            title='Trade Returns Distribution',
            xaxis_title='Return (%)',
            yaxis_title='Count',
            template='plotly_dark'
        )
        return fig
    
    # Extract returns from trades
    returns = [trade['profit_pct'] for trade in trades]
    
    # Create a histogram
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=30,
        marker=dict(
            color='blue',
            line=dict(
                color='white',
                width=0.5
            )
        ),
        opacity=0.75
    ))
    
    # Add vertical line at zero
    fig.add_shape(
        type="line",
        x0=0, y0=0,
        x1=0, y1=1,
        yref='paper',
        line=dict(
            color="red",
            width=2,
            dash="dash"
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Trade Returns Distribution',
        xaxis_title='Return (%)',
        yaxis_title='Count',
        bargap=0.1,
        template='plotly_dark'
    )
    
    return fig