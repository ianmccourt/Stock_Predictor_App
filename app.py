import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
from SimpleStockPredictor import SimpleStockPredictor

# Cache for data fetching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    return yf.download(symbol, period=period, progress=False)

# Validate ticker
@st.cache_data(ttl=3600)
def validate_ticker(symbol: str) -> bool:
    try:
        data = yf.download(symbol, period="1d", progress=False)
        return len(data) > 0
    except Exception:
        return False

# Page config
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Stock Price Predictor")
st.markdown("""
This app uses machine learning to predict stock prices and provide trading insights.
""")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    symbol = st.text_input("Stock Symbol", "AAPL").upper().strip()
    
    # Validate symbol format
    if not symbol.replace(".", "").replace("-", "").isalnum():
        st.warning("Please enter a valid stock symbol")
        st.stop()
    
    # Validate ticker exists
    with st.spinner("Validating ticker..."):
        if not validate_ticker(symbol):
            st.error(f"No data found for {symbol}. Please check the ticker symbol.")
            st.markdown("""
            **Valid Examples:**
            - US Stocks: AAPL, MSFT, GOOGL
            - International: VOD.L, RY.TO, BMW.F
            - ETFs: SPY, QQQ
            - Crypto: BTC-USD, ETH-USD
            """)
            st.stop()
        else:
            st.success(f"Valid ticker: {symbol}")
    
    timeframe = st.selectbox(
        "Timeframe",
        ['1d', '1h', '5m'],
        index=0,
        help="Select prediction timeframe"
    )
    
    predict_button = st.button("Predict", type="primary")

# Main content
if predict_button:
    try:
        with st.spinner("Initializing prediction model..."):
            predictor = SimpleStockPredictor()
        
        # Training progress
        with st.status("Making prediction...") as status:
            st.write("Training model...")
            if predictor.train(symbol, timeframe):
                st.write("Getting prediction...")
                prediction = predictor.predict(symbol, timeframe)
                st.write("Calculating metrics...")
                metrics = predictor.evaluate_performance(symbol, timeframe)
                status.update(label="Prediction complete!", state="complete")
            else:
                st.error("Failed to train model. Please try again.")
                st.stop()

        # Display results in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Current Price",
                value=f"${prediction['current_price']:.2f}"
            )
            
        with col2:
            st.metric(
                label="Predicted Price",
                value=f"${prediction['predicted_price']:.2f}",
                delta=f"{prediction['predicted_return']*100:.1f}%"
            )
            
        with col3:
            st.metric(
                label="Confidence Score",
                value=f"{prediction['confidence_score']:.1f}%"
            )

        # Performance Metrics
        st.subheader("Performance Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            metrics_df = pd.DataFrame({
                'Metric': [
                    'Directional Accuracy',
                    'Win Rate',
                    'Sharpe Ratio',
                    'Sortino Ratio',
                    'Maximum Drawdown'
                ],
                'Value': [
                    f"{metrics['directional_accuracy']:.1%}",
                    f"{metrics['win_rate']:.1%}",
                    f"{metrics['sharpe_ratio']:.2f}",
                    f"{metrics['sortino_ratio']:.2f}",
                    f"{metrics['max_drawdown']:.1%}"
                ]
            })
            st.dataframe(metrics_df, hide_index=True)
            
        with col2:
            metrics_df2 = pd.DataFrame({
                'Metric': [
                    'Calmar Ratio',
                    'Information Ratio',
                    'Beta',
                    'Annualized Volatility',
                    'Annualized Return'
                ],
                'Value': [
                    f"{metrics['calmar_ratio']:.2f}",
                    f"{metrics['information_ratio']:.2f}",
                    f"{metrics['beta']:.2f}",
                    f"{metrics['annualized_volatility']:.1%}",
                    f"{metrics['annualized_return']:.1%}"
                ]
            })
            st.dataframe(metrics_df2, hide_index=True)

        # Historical Data Plot
        st.subheader("Historical Price Data")

        try:
            # Debug prints
            st.write("Fetching data...")
            
            # Fetch historical data with explicit parameters
            df = yf.download(
                tickers=symbol,
                period="1y",
                interval="1d",
                auto_adjust=True,
                progress=False
            )
            
            # Debug information
            st.write(f"Data shape: {df.shape}")
            st.write(f"Column names: {df.columns.tolist()}")
            
            # Ensure column names are correct for plotly
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            if not df.empty:
                # Create figure
                fig = go.Figure()
                
                # Add candlestick trace
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name=symbol,
                    increasing_line_color='#26A69A',  # Green
                    decreasing_line_color='#EF5350'   # Red
                ))
                
                # Add moving averages
                ma20 = df['Close'].rolling(window=20).mean()
                ma50 = df['Close'].rolling(window=50).mean()
                
                fig.add_trace(go.Scatter(
                    x=df.index, 
                    y=ma20, 
                    name='20-day MA',
                    line=dict(color='#2962FF', width=1)
                ))
                
                fig.add_trace(go.Scatter(
                    x=df.index, 
                    y=ma50, 
                    name='50-day MA',
                    line=dict(color='#FF6D00', width=1)
                ))
                
                # Update layout with better defaults
                fig.update_layout(
                    title=dict(
                        text=f'{symbol} Stock Price (Past Year)',
                        x=0.5,  # Center title
                        font=dict(size=24)
                    ),
                    yaxis_title='Stock Price (USD)',
                    xaxis_title='Date',
                    template='plotly_dark',
                    xaxis_rangeslider_visible=False,
                    height=600,
                    yaxis=dict(
                        tickprefix="$",
                        tickformat=".2f",
                        side='right'  # Move price axis to right side
                    ),
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor='rgba(0,0,0,0.5)'
                    ),
                    margin=dict(t=100, l=50, r=50, b=50)
                )
                
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Display data summary
                with st.expander("Show Raw Data"):
                    # Calculate daily returns
                    daily_returns = df['Close'].pct_change()
                    
                    # Summary metrics
                    summary = pd.DataFrame({
                        'Metric': [
                            'Current Price',
                            'Daily Change',
                            'Total Trading Days',
                            'Period High',
                            'Period Low',
                            'Average Volume',
                            'Average Daily Return',
                            'Return Volatility'
                        ],
                        'Value': [
                            f"${df['Close'].iloc[-1]:.2f}",
                            f"{(df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1):.2%}",
                            len(df),
                            f"${df['High'].max():.2f}",
                            f"${df['Low'].min():.2f}",
                            f"{df['Volume'].mean():,.0f}",
                            f"{daily_returns.mean():.2%}",
                            f"{daily_returns.std():.2%}"
                        ]
                    })
                    
                    st.dataframe(summary, hide_index=True)
                    st.write(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
                    
                    # Show recent data
                    st.write("\nRecent Price Data:")
                    st.dataframe(df.tail())

            else:
                st.warning(f"No historical data available for {symbol}")
                
        except Exception as e:
            st.error(f"Error fetching historical data: {str(e)}")
            st.exception(e)  # This will show the full traceback

        # Feature Importance
        if 'top_features' in prediction:
            st.subheader("Top Predictive Features")
            features_df = pd.DataFrame(prediction['top_features'])
            st.dataframe(features_df, hide_index=True)

    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.exception(e)  # This will show the full traceback in development

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Data from Yahoo Finance")

# Add cache management
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.success("Cache cleared!") 