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
# 📈 Tesla and BYD Stock Analytics

Real-time stock price data from Yahoo Finance with correlation analysis.
'''

# Add some spacing
''
''

# Date range selector
end_date = datetime.now().date()
start_date = end_date - timedelta(days=365)  # Default to last year

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input('Start Date', start_date)
with col2:
    end_date = st.date_input('End Date', end_date)

# Fetch data
tickers = ['TSLA', 'BYD']
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
    correlation = returns_df['TSLA'].corr(returns_df['BYD'])
    st.metric(label="Correlation Coefficient (Daily Returns)", value=f"{correlation:.3f}")

    # Scatter plot of returns
    st.subheader('Scatter Plot of Daily Returns')
    fig = px.scatter(returns_df, x='TSLA', y='BYD', 
                     title='Tesla vs BYD Daily Returns',
                     labels={'TSLA': 'Tesla Daily Return', 'BYD': 'BYD Daily Return'})
    fig.add_trace(go.Scatter(x=[returns_df['TSLA'].min(), returns_df['TSLA'].max()], 
                             y=[returns_df['TSLA'].min(), returns_df['TSLA'].max()], 
                             mode='lines', name='45° Line', line=dict(dash='dash')))
    st.plotly_chart(fig)

    # Rolling correlation
    st.subheader('Rolling Correlation (30-day window)')
    rolling_corr = returns_df['TSLA'].rolling(window=30).corr(returns_df['BYD'])
    st.line_chart(rolling_corr)
