import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Stock Analytics Dashboard',
    page_icon='📈',  # Stock chart emoji
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_stock_data(tickers, start_date, end_date):
    """Grab stock data from Yahoo Finance.

    This uses caching to avoid having to fetch data every time.
    """
    data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        data[ticker] = hist['Close']
    df = pd.DataFrame(data)
    df.index = df.index.date  # Convert to date for easier handling
    return df

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# 📈 Stock Analytics Dashboard

Real-time stock price data from Yahoo Finance with correlation analysis.
'''

# Add some spacing
''
''

# Stock ticker inputs
col1, col2 = st.columns(2)
with col1:
    ticker1 = st.text_input('First Stock Ticker', 'TSLA')
with col2:
    ticker2 = st.text_input('Second Stock Ticker', 'BYD')

tickers = [ticker1.upper(), ticker2.upper()]

if not ticker1 or not ticker2:
    st.error("Please enter both stock tickers.")
    st.stop()
elif ticker1.upper() == ticker2.upper():
    st.error("Please enter two different stock tickers.")
    st.stop()

# Date range selector
end_date = datetime.now().date()
start_date = end_date - timedelta(days=365)  # Default to last year

col3, col4 = st.columns(2)
with col3:
    start_date = st.date_input('Start Date', start_date)
with col4:
    end_date = st.date_input('End Date', end_date)
stock_df = get_stock_data(tickers, start_date, end_date)

if stock_df.empty:
    st.error("No data available for the selected date range.")
else:
    # Display stock prices over time
    st.header('Stock Prices Over Time', divider='gray')
    st.line_chart(stock_df)

    # Calculate daily returns
    returns_df = stock_df.pct_change().dropna()

    # Correlation analysis
    st.header('Correlation Analysis', divider='gray')

    # Correlation coefficient
    correlation = returns_df[ticker1.upper()].corr(returns_df[ticker2.upper()])
    st.metric(label=f"Correlation Coefficient (Daily Returns) - {ticker1.upper()} vs {ticker2.upper()}", value=f"{correlation:.3f}")

    # Scatter plot of returns
    st.subheader(f'Scatter Plot of Daily Returns: {ticker1.upper()} vs {ticker2.upper()}')
    fig = px.scatter(returns_df, x=ticker1.upper(), y=ticker2.upper(), 
                     title=f'{ticker1.upper()} vs {ticker2.upper()} Daily Returns',
                     labels={ticker1.upper(): f'{ticker1.upper()} Daily Return', ticker2.upper(): f'{ticker2.upper()} Daily Return'})
    fig.add_trace(go.Scatter(x=[returns_df[ticker1.upper()].min(), returns_df[ticker1.upper()].max()], 
                             y=[returns_df[ticker1.upper()].min(), returns_df[ticker1.upper()].max()], 
                             mode='lines', name='45° Line', line=dict(dash='dash')))
    st.plotly_chart(fig)

    # Rolling correlation
    st.subheader(f'Rolling Correlation (30-day window): {ticker1.upper()} vs {ticker2.upper()}')
    rolling_corr = returns_df[ticker1.upper()].rolling(window=30).corr(returns_df[ticker2.upper()])
    st.line_chart(rolling_corr)
