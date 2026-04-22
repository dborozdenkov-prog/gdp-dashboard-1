import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Financial Analytics Dashboard',
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

@st.cache_data
def get_fx_data(tickers, start_date, end_date):
    """Grab FX data from Yahoo Finance.

    This uses caching to avoid having to fetch data every time.
    """
    data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date, interval='1h')
        data[ticker] = hist['Close']
    df = pd.DataFrame(data)
    return df

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# 📈 Financial Analytics Dashboard

Stock and FX analysis with correlation and Fourier Transform.
'''

# Add some spacing
''
''

# Create tabs
tab1, tab2 = st.tabs(["Stock Analytics", "FX Analytics"])

with tab1:
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

with tab2:
    # FX pair input
    fx_pair = st.text_input('FX Pair', 'EURUSD=X')

    if not fx_pair:
        st.error("Please enter an FX pair.")
        st.stop()

    # Date range selector for FX
    end_date_fx = datetime.now().date()
    start_date_fx = end_date_fx - timedelta(days=30)  # Default to last 30 days for hourly data

    col5, col6 = st.columns(2)
    with col5:
        start_date_fx = st.date_input('Start Date (FX)', start_date_fx)
    with col6:
        end_date_fx = st.date_input('End Date (FX)', end_date_fx)
    fx_df = get_fx_data([fx_pair], start_date_fx, end_date_fx)

    if fx_df.empty:
        st.error("No data available for the selected date range.")
    else:
        # Display FX prices over time
        st.header('FX Prices Over Time', divider='gray')
        fig_fx = px.line(fx_df, x=fx_df.index, y=fx_pair, title=f'{fx_pair} Prices Over Time')
        fig_fx.update_layout(height=400, xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_fx)

        # Fourier Transform Analysis
        st.header('Fourier Transform Analysis', divider='gray')

        prices = fx_df[fx_pair].values

        # Compute FFT
        fft = np.fft.fft(prices)
        freqs = np.fft.fftfreq(len(prices), d=1)  # d=1 for hourly data
        magnitude = np.abs(fft)

        # Plot power spectrum (positive frequencies only)
        positive_freqs = freqs[:len(freqs)//2 + 1]
        positive_magnitude = magnitude[:len(magnitude)//2 + 1]
        power_spectrum_df = pd.DataFrame({
            'Frequency': positive_freqs,
            'Magnitude': positive_magnitude
        })
        fig = px.line(power_spectrum_df, x='Frequency', y='Magnitude', title='Power Spectrum')
        fig.update_yaxes(type="log")
        fig.update_layout(height=400, xaxis_title='Frequency (cycles per hour)', yaxis_title='Magnitude (log scale)')
        st.plotly_chart(fig)

        # Denoise by filtering out high-frequency components
        threshold = np.percentile(magnitude, 90)  # Retain top 10% of components
        fft_filtered = fft.copy()
        fft_filtered[magnitude < threshold] = 0
        smoothed_prices = np.fft.ifft(fft_filtered).real

        # Predict next hour price using linear extrapolation on the last 24 smoothed prices
        last_points = smoothed_prices[-24:]
        x = np.arange(len(last_points))
        slope, intercept = np.polyfit(x, last_points, 1)
        next_price = slope * len(last_points) + intercept

        st.metric(label=f"Predicted Next Hour Price for {fx_pair}", value=f"{next_price:.4f}")

        # Display smoothed prices chart
        st.subheader('Smoothed Prices (Denoised via FFT)')
        smoothed_df = pd.DataFrame({'Smoothed Price': smoothed_prices}, index=fx_df.index)
        fig_smoothed = px.line(smoothed_df, x=smoothed_df.index, y='Smoothed Price', title='Smoothed Prices')
        fig_smoothed.update_layout(height=400, xaxis_title='Date', yaxis_title='Smoothed Price')
        st.plotly_chart(fig_smoothed)
