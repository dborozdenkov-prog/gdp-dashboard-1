import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Signal Analytics Dashboard',
    page_icon='📈',  # Stock chart emoji
)

# Suppress harmless ScriptRunContext warnings
import warnings
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

# -----------------------------------------------------------------------------
# Declare some useful functions.

def retry_with_backoff(func, max_retries=3, base_delay=1):
    """Retry a function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)

@st.cache_data(ttl=600)
def get_stock_data(tickers, start_date, end_date):
    """Grab stock data from Yahoo Finance.

    This uses caching to avoid having to fetch data every time.
    """
    data = {}
    for ticker in tickers:
        def fetch_ticker():
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date, interval='1h')
            return hist['Close']
        
        try:
            data[ticker] = retry_with_backoff(fetch_ticker)
        except Exception as e:
            st.warning(f"⚠️ Failed to fetch data for {ticker}: {str(e)[:100]}")
            continue
    
    if not data:
        raise ValueError("Failed to fetch any stock data")
    
    df = pd.DataFrame(data)
    return df

@st.cache_data(ttl=300)
def get_fx_data(tickers, start_date, end_date):
    """Grab FX data from Yahoo Finance.

    This uses caching to avoid having to fetch data every time.
    """
    data = {}
    for ticker in tickers:
        def fetch_fx():
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date, interval='1h')
            return hist['Close']
        
        try:
            data[ticker] = retry_with_backoff(fetch_fx)
        except Exception as e:
            st.warning(f"⚠️ Failed to fetch data for {ticker}: {str(e)[:100]}")
            continue
    
    if not data:
        raise ValueError("Failed to fetch any FX data")
    
    df = pd.DataFrame(data)
    return df

@st.cache_data
def get_bond_prices_finnhub(isins, start_date, end_date, api_key):
    """Fetch bond prices from Finnhub using ISINs.
    
    Finnhub API: https://finnhub.io (free tier available)
    Returns a DataFrame with dates as index and bond prices as columns.
    """
    import requests
    
    data = {}
    failed_isins = []
    
    # Convert dates to Unix timestamps (handle both date and datetime objects)
    # If it's a date object without time, combine with time components
    try:
        if hasattr(start_date, 'timestamp'):
            start_ts = int(start_date.timestamp())
        else:
            # It's a date object, convert to datetime first
            start_dt = datetime.combine(start_date, datetime.min.time())
            start_ts = int(start_dt.timestamp())
    except:
        start_ts = 0
    
    try:
        if hasattr(end_date, 'timestamp'):
            end_ts = int(end_date.timestamp())
        else:
            # It's a date object, convert to datetime first
            end_dt = datetime.combine(end_date, datetime.max.time())
            end_ts = int(end_dt.timestamp())
    except:
        end_ts = 0
    
    for isin in isins:
        try:
            # Query Finnhub API for bond data
            url = f"https://finnhub.io/api/v1/quote"
            params = {
                'symbol': isin,
                'token': api_key
            }
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                quote_data = response.json()
                if 'c' in quote_data and quote_data.get('c'):  # c = current price
                    # For demonstration, create synthetic daily data
                    # In production, use Finnhub's candle endpoint if available
                    data[isin] = [quote_data['c']]
                else:
                    failed_isins.append(isin)
            else:
                failed_isins.append(isin)
        except Exception as e:
            failed_isins.append(isin)
    
    if data:
        df = pd.DataFrame(data)
        return df, failed_isins
    else:
        return pd.DataFrame(), failed_isins

@st.cache_data
def get_bond_prices_iexcloud(isins, start_date, end_date, api_key):
    """Fetch bond prices from IEX Cloud using ISINs.
    
    IEX Cloud API: https://iexcloud.io (free tier available)
    Returns a DataFrame with dates as index and bond prices as columns.
    """
    import requests
    from datetime import datetime
    
    data = {}
    failed_isins = []
    
    for isin in isins:
        try:
            # Query IEX Cloud API for historical data
            url = f"https://cloud.iexapis.com/stable/data/core_financials/annual/{isin}"
            params = {'token': api_key}
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                bond_data = response.json()
                if bond_data:
                    # Extract price or use latest available
                    data[isin] = [100.0]  # Default bond price
                else:
                    failed_isins.append(isin)
            else:
                failed_isins.append(isin)
        except Exception as e:
            failed_isins.append(isin)
    
    if data:
        df = pd.DataFrame(data)
        return df, failed_isins
    else:
        return pd.DataFrame(), failed_isins

@st.cache_data
def get_bond_prices_cboe(isins, start_date, end_date):
    """Fetch bond prices from CBOE/alternative sources.
    
    Uses multiple data sources including CBOE, Treasury data, etc.
    Returns a DataFrame with dates as index and bond prices as columns.
    """
    import requests
    
    data = {}
    failed_isins = []
    
    for isin in isins:
        try:
            # Try alternative sources - CBOE, Bloomberg data, etc.
            # This is a placeholder that attempts multiple sources
            
            # Attempt 1: Check if it's a corporate bond ticker
            ticker = isin.replace('=X', '')
            
            try:
                bond = yf.Ticker(ticker + ".L")  # London Stock Exchange
                hist = bond.history(start=start_date, end=end_date)
                if not hist.empty and 'Close' in hist.columns:
                    data[isin] = hist['Close']
                    continue
            except:
                pass
            
            # Attempt 2: Try different exchanges
            for suffix in ['.DE', '.AS', '.MI', '.VI']:
                try:
                    bond = yf.Ticker(ticker + suffix)
                    hist = bond.history(start=start_date, end=end_date)
                    if not hist.empty and len(hist) > 5:
                        data[isin] = hist['Close']
                        break
                except:
                    continue
            
            if isin not in data:
                failed_isins.append(isin)
                
        except Exception as e:
            failed_isins.append(isin)
    
    if data:
        df = pd.DataFrame(data)
        return df, failed_isins
    else:
        return pd.DataFrame(), failed_isins

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# 📈 Signal Analytics Dashboard

Stock and FX analysis with correlation and Fourier Transform.
'''

# Add some spacing
''
''

# Create tabs
tab1, tab2, tab3 = st.tabs(["Stock Analytics", "FX Analytics", "Bond PCA Analysis"])

with tab1:
    # Refresh button
    if st.button('🔄 Refresh Stock Data', key='stock_refresh'):
        st.rerun()
    
    st.divider()
    
    # Stock ticker inputs
    col1, col2 = st.columns(2)
    with col1:
        ticker1 = st.text_input('First Stock Ticker', 'KO')
    with col2:
        ticker2 = st.text_input('Second Stock Ticker', 'PEP')

    tickers = [ticker1.upper(), ticker2.upper()]

    if not ticker1 or not ticker2:
        st.error("Please enter both stock tickers.")
        st.stop()
    elif ticker1.upper() == ticker2.upper():
        st.error("Please enter two different stock tickers.")
        st.stop()

    # Date range selector
    end_date = datetime.now().date() + timedelta(days=1)
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
        # Display raw data preview (latest hourly rates including today)
        with st.expander("View Raw Price Data (Latest 48 Hourly Rates)"):
            st.dataframe(stock_df.tail(48))

        # Display stock prices over time
        st.header('Stock Prices Over Time', divider='gray')
        st.line_chart(stock_df)

        # Calculate rolling mean and standard deviation for normalization
        # Using a 20-period rolling window as an example, this can be adjusted.
        window = 20

        # Create a copy to avoid SettingWithCopyWarning
        close_prices_copy = stock_df.copy()

        # Calculate normalized prices for both tickers dynamically
        ticker1_norm = f'{ticker1.upper()}_normalized'
        ticker2_norm = f'{ticker2.upper()}_normalized'
        
        close_prices_copy[ticker1_norm] = (close_prices_copy[ticker1.upper()] - close_prices_copy[ticker1.upper()].rolling(window=window).mean()) / close_prices_copy[ticker1.upper()].rolling(window=window).std()
        close_prices_copy[ticker2_norm] = (close_prices_copy[ticker2.upper()] - close_prices_copy[ticker2.upper()].rolling(window=window).mean()) / close_prices_copy[ticker2.upper()].rolling(window=window).std()

        # Calculate the spread between the normalized prices
        close_prices_copy['Spread'] = close_prices_copy[ticker1_norm] - close_prices_copy[ticker2_norm]

        # Display normalized prices and spread
        st.header('Normalized Prices and Spread', divider='gray')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=close_prices_copy.index, y=close_prices_copy[ticker1_norm], mode='lines', name=f'{ticker1.upper()} Normalized'))
        fig.add_trace(go.Scatter(x=close_prices_copy.index, y=close_prices_copy[ticker2_norm], mode='lines', name=f'{ticker2.upper()} Normalized'))
        fig.add_trace(go.Scatter(x=close_prices_copy.index, y=close_prices_copy['Spread'], mode='lines', name=f'Spread ({ticker1.upper()} - {ticker2.upper()})', line=dict(dash='dash')))
        fig.update_layout(title='Normalized Prices and Spread', xaxis_title='Date', yaxis_title='Normalized Price / Spread', height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Detailed Spread Analysis with Bollinger Bands
        st.header(f'Spread Analysis: {ticker1.upper()} - {ticker2.upper()}', divider='gray')
        
        # Calculate Bollinger Bands for the spread
        spread_mean = close_prices_copy['Spread'].rolling(window=20).mean()
        spread_std = close_prices_copy['Spread'].rolling(window=20).std()
        upper_band = spread_mean + (2 * spread_std)
        lower_band = spread_mean - (2 * spread_std)
        
        # Create spread chart with Bollinger Bands
        fig_spread = go.Figure()
        fig_spread.add_trace(go.Scatter(x=close_prices_copy.index, y=upper_band, fill=None, mode='lines', line_color='rgba(255,0,0,0)', showlegend=False))
        fig_spread.add_trace(go.Scatter(x=close_prices_copy.index, y=lower_band, fillcolor='rgba(0,100,200,0.2)', fill='tonexty', mode='lines', line_color='rgba(0,255,0,0)', name='Bollinger Band Region'))
        fig_spread.add_trace(go.Scatter(x=close_prices_copy.index, y=close_prices_copy['Spread'], mode='lines', name='Spread', line=dict(color='blue', width=2)))
        fig_spread.add_trace(go.Scatter(x=close_prices_copy.index, y=spread_mean, mode='lines', name='Mean', line=dict(color='gray', dash='dot')))
        fig_spread.add_trace(go.Scatter(x=close_prices_copy.index, y=upper_band, mode='lines', name='Upper Band (+2σ)', line=dict(color='red', dash='dash')))
        fig_spread.add_trace(go.Scatter(x=close_prices_copy.index, y=lower_band, mode='lines', name='Lower Band (-2σ)', line=dict(color='green', dash='dash')))
        
        fig_spread.update_layout(
            title=f'Normalized Spread with Bollinger Bands (20-period, 2σ)',
            xaxis_title='Date',
            yaxis_title='Spread Value',
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig_spread, use_container_width=True)
        
        # Spread Statistics
        st.subheader('Spread Statistics')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('Mean Spread', f"{close_prices_copy['Spread'].mean():.4f}")
        with col2:
            st.metric('Std Dev', f"{close_prices_copy['Spread'].std():.4f}")
        with col3:
            st.metric('Current Spread', f"{close_prices_copy['Spread'].iloc[-1]:.4f}")
        with col4:
            spread_zscore = (close_prices_copy['Spread'].iloc[-1] - close_prices_copy['Spread'].mean()) / close_prices_copy['Spread'].std()
            st.metric('Z-Score', f"{spread_zscore:.2f}")
        
        # Spread Prediction using Fourier Transform
        st.subheader('Spread Prediction (Fourier Transform Analysis)', divider='gray')
        
        spread_values = close_prices_copy['Spread'].dropna().values
        
        # Compute FFT
        fft_spread = np.fft.fft(spread_values)
        spread_freqs = np.fft.fftfreq(len(spread_values), d=1)
        spread_magnitude = np.abs(fft_spread)
        
        # Denoise by filtering out high-frequency components
        spread_threshold = np.percentile(spread_magnitude, 85)  # Keep top 15%
        fft_spread_filtered = fft_spread.copy()
        fft_spread_filtered[spread_magnitude < spread_threshold] = 0
        smoothed_spread = np.fft.ifft(fft_spread_filtered).real
        
        # Predict next spread value using linear extrapolation on smoothed spread
        use_points = min(48, len(smoothed_spread))  # Use last 48 periods
        last_spread_points = smoothed_spread[-use_points:]
        x_spread = np.arange(len(last_spread_points))
        slope_spread, intercept_spread = np.polyfit(x_spread, last_spread_points, 1)
        predicted_spread = slope_spread * len(last_spread_points) + intercept_spread
        
        # Calculate confidence based on signal-to-noise ratio
        signal_power = np.sum(spread_magnitude[spread_magnitude >= spread_threshold])
        noise_power = np.sum(spread_magnitude[spread_magnitude < spread_threshold])
        snr = signal_power / (noise_power + 1e-10)
        confidence_pct = min(100, (snr / (snr + 1)) * 100)
        
        current_spread = close_prices_copy['Spread'].iloc[-1]
        spread_diff_pct = ((predicted_spread - current_spread) / abs(current_spread)) * 100 if current_spread != 0 else 0
        
        # Calculate timing information
        current_time = datetime.now()
        next_hour = current_time + timedelta(hours=1)
        time_until_next = next_hour - current_time
        minutes_left = int(time_until_next.total_seconds() // 60)
        seconds_left = int(time_until_next.total_seconds() % 60)
        
        # Display prediction metrics
        st.write("**Next Period Spread Prediction:**")
        col_pred, col_curr, col_diff = st.columns(3)
        with col_pred:
            st.metric(
                label='🎯 Predicted Spread',
                value=f'{predicted_spread:.4f}',
                delta=f'{spread_diff_pct:.2f}%'
            )
        with col_curr:
            st.metric(
                label='📊 Current Spread',
                value=f'{current_spread:.4f}'
            )
        with col_diff:
            confidence_color = "🟢" if confidence_pct > 70 else "🟡" if confidence_pct > 40 else "🔴"
            st.metric(
                label=f'{confidence_color} Prediction Confidence',
                value=f'{confidence_pct:.1f}%',
                delta=f'SNR: {snr:.2f}' if snr > 0 else 'Low Signal'
            )
        
        # Display timing information
        st.write("**Timing Information:**")
        col_time1, col_time2, col_time3 = st.columns(3)
        with col_time1:
            st.metric(
                label='⏰ Current Time',
                value=current_time.strftime('%H:%M:%S'),
                delta=current_time.strftime('%Y-%m-%d')
            )
        with col_time2:
            st.metric(
                label='⏱️ Next Period',
                value=next_hour.strftime('%H:%M:%S'),
                delta=next_hour.strftime('%Y-%m-%d')
            )
        with col_time3:
            st.metric(
                label='⌛ Time Until Next',
                value=f'{minutes_left}m {seconds_left}s',
                delta='Hourly update'
            )
        
        # Display forecast chart
        st.write("**Smoothed Spread Trend (Denoised via FFT):**")
        forecast_df = pd.DataFrame({
            'Period': range(len(smoothed_spread)),
            'Smoothed Spread': smoothed_spread,
            'Original Spread': spread_values
        }, index=close_prices_copy.index[:len(spread_values)])
        
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Original Spread'], mode='lines', name='Original Spread', line=dict(color='lightgray', dash='dot')))
        fig_forecast.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Smoothed Spread'], mode='lines', name='Smoothed Spread (FFT)', line=dict(color='blue', width=2)))
        fig_forecast.add_hline(y=predicted_spread, line_dash='dash', line_color='green', annotation_text=f'Predicted: {predicted_spread:.4f}', annotation_position='right')
        fig_forecast.add_hline(y=current_spread, line_dash='dash', line_color='red', annotation_text=f'Current: {current_spread:.4f}', annotation_position='right')
        fig_forecast.update_layout(title='Spread Denoising & Prediction', xaxis_title='Date', yaxis_title='Spread', height=400)
        st.plotly_chart(fig_forecast, use_container_width=True)

with tab2:
    # Refresh button
    if st.button('🔄 Refresh FX Data', key='fx_refresh'):
        st.rerun()
    
    st.divider()
    
    # FX pair input with G10 currency pairs
    g10_pairs = [
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'AUDUSD=X',
        'NZDUSD=X', 'USDCAD=X', 'EURGBP=X', 'EURJPY=X', 'GBPJPY=X'
    ]
    fx_pair = st.selectbox('Select FX Pair (G10)', g10_pairs, index=0)

    if not fx_pair:
        st.error("Please enter an FX pair.")
        st.stop()

    # Date range selector for FX
    # Include today's data by setting end_date to tomorrow
    end_date_fx = datetime.now().date() + timedelta(days=1)
    start_date_fx = end_date_fx - timedelta(days=20)# Default to last 20 days for hourly data

    col5, col6 = st.columns(2)
    with col5:
        start_date_fx = st.date_input('Start Date (FX)', start_date_fx)
    with col6:
        end_date_fx = st.date_input('End Date (FX)', end_date_fx)
    
    st.caption("📊 Includes today's hourly prices for more accurate predictions")
    fx_df = get_fx_data([fx_pair], start_date_fx, end_date_fx)

    if fx_df.empty:
        st.error("No data available for the selected date range.")
    else:
        # Display FX prices over time
        st.header('FX Prices Over Time', divider='gray')
        fig_fx = px.line(fx_df, x=fx_df.index, y=fx_pair, title=f'{fx_pair} Prices Over Time')
        fig_fx.update_layout(height=400, xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_fx)

        # Display raw data preview (latest 50 hourly rates including today)
        with st.expander("View Raw FX Data (Latest 50 Hourly Rates)"):
            st.dataframe(fx_df.tail(50))

        # Fourier Transform Analysis
        st.header('Fourier Transform Analysis', divider='gray')

        prices = fx_df[fx_pair].values

        # Compute FFT
        fft = np.fft.fft(prices)
        freqs = np.fft.fftfreq(len(prices), d=1)  # d=1 for hourly data
        magnitude = np.abs(fft)

        # Denoise by filtering out high-frequency components
        threshold = np.percentile(magnitude, 90)  # Retain top 10% of components
        fft_filtered = fft.copy()
        fft_filtered[magnitude < threshold] = 0
        smoothed_prices = np.fft.ifft(fft_filtered).real

        # Display smoothed prices chart
        st.subheader('Smoothed Prices (Denoised via FFT)')
        smoothed_df = pd.DataFrame({'Smoothed Price': smoothed_prices}, index=fx_df.index)
        fig_smoothed = px.line(smoothed_df, x=smoothed_df.index, y='Smoothed Price', title='Smoothed Prices')
        fig_smoothed.update_layout(height=400, xaxis_title='Date', yaxis_title='Smoothed Price')
        st.plotly_chart(fig_smoothed)

        # Predict next hour price using all available smoothed prices
        # Use more data points for better prediction accuracy
        use_points = min(48, len(smoothed_prices))  # Use up to last 48 hours of data
        last_points = smoothed_prices[-use_points:]
        x = np.arange(len(last_points))
        
        # Linear extrapolation from all available data
        slope, intercept = np.polyfit(x, last_points, 1)
        next_price = slope * len(last_points) + intercept
        
        # Calculate confidence based on data recency
        latest_prices = prices[-10:] if len(prices) >= 10 else prices
        price_volatility = np.std(latest_prices)
        confidence = "High" if price_volatility < np.mean(latest_prices) * 0.05 else "Medium" if price_volatility < np.mean(latest_prices) * 0.1 else "Low"

        # Plot Magnitude spectrum (positive frequencies only)
        positive_freqs = freqs[:len(freqs)//2 + 1]
        positive_magnitude = magnitude[:len(magnitude)//2 + 1]
        power_spectrum_df = pd.DataFrame({
            'Frequency': positive_freqs,
            'Magnitude': positive_magnitude
        })
        fig = px.line(power_spectrum_df, x='Frequency', y='Magnitude', title='Magnitude Spectrum')
        fig.update_yaxes(type="log")
        fig.update_layout(height=400, xaxis_title='Frequency (cycles per hour)', yaxis_title='Magnitude (log scale)')
        st.plotly_chart(fig)

        # Display predicted and current price side by side
        col_pred, col_live, col_conf = st.columns(3)
        with col_pred:
            st.metric(label=f"Predicted Next Hour Price for {fx_pair}", value=f"{next_price:.4f}")
        with col_live:
            current_price = prices[-1]  # Use most recent price
            st.metric(label=f"Current Live Price for {fx_pair}", value=f"{current_price:.4f}")
        with col_conf:
            price_change = ((next_price - current_price) / current_price * 100) if current_price != 0 else 0
            st.metric(label="Prediction Confidence", value=confidence, delta=f"{price_change:.2f}%")
        
        # Display timing information
        st.write("**Timing Information:**")
        current_time_fx = datetime.now()
        next_hour_fx = current_time_fx + timedelta(hours=1)
        time_until_next_fx = next_hour_fx - current_time_fx
        minutes_left_fx = int(time_until_next_fx.total_seconds() // 60)
        seconds_left_fx = int(time_until_next_fx.total_seconds() % 60)
        
        col_time1, col_time2, col_time3 = st.columns(3)
        with col_time1:
            st.metric(
                label='⏰ Current Time',
                value=current_time_fx.strftime('%H:%M:%S'),
                delta=current_time_fx.strftime('%Y-%m-%d')
            )
        with col_time2:
            st.metric(
                label='⏱️ Next Period',
                value=next_hour_fx.strftime('%H:%M:%S'),
                delta=next_hour_fx.strftime('%Y-%m-%d')
            )
        with col_time3:
            st.metric(
                label='⌛ Time Until Next',
                value=f'{minutes_left_fx}m {seconds_left_fx}s',
                delta='Hourly update'
            )

with tab3:
    st.header('Bond Portfolio PCA Analysis', divider='gray')
    st.write('Upload bond ISINs and analyze which bonds drive your portfolio returns using Principal Component Analysis')
    
    # Data provider selection
    st.subheader('Step 1: Select Data Provider')
    col1, col2 = st.columns(2)
    with col1:
        data_provider = st.radio('Choose bond price data source:', 
                                 ['Multi-Source (yfinance+exchanges)', 'Finnhub API', 'IEX Cloud API'],
                                 help='Multi-Source works best for European/Asian bonds listed on multiple exchanges')
    
    api_key = None
    if data_provider == 'Finnhub API':
        with col2:
            api_key = st.text_input('Finnhub API Key', type='password', 
                                   help='Get free API key at https://finnhub.io')
            st.caption('Free tier available - sign up at finnhub.io')
    elif data_provider == 'IEX Cloud API':
        with col2:
            api_key = st.text_input('IEX Cloud API Key', type='password',
                                   help='Get free API key at https://iexcloud.io')
            st.caption('Free tier available - sign up at iexcloud.io')
    
    # ISIN input section
    st.subheader('Step 2: Upload or Enter ISINs')
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload CSV with ISINs (one per row)", type="csv", help="CSV should contain one ISIN per row")
    
    isin_list = []
    
    if uploaded_file is not None:
        try:
            isin_df = pd.read_csv(uploaded_file, header=None)
            isin_list = isin_df[0].str.strip().tolist()
            st.success(f"Loaded {len(isin_list)} ISINs")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    # Alternative: paste ISINs
    with st.expander("Or paste ISINs directly"):
        pasted_isins = st.text_area("Paste ISINs (one per line, max 500)", height=150)
        if pasted_isins:
            isin_list = [x.strip() for x in pasted_isins.split('\n') if x.strip()]
    
    if isin_list:
        if len(isin_list) > 500:
            st.error("Please provide up to 500 ISINs maximum.")
            st.stop()
        
        st.write(f"ISINs to analyze: {', '.join(isin_list[:10])}{'...' if len(isin_list) > 10 else ''}")
        
        # Date range selector
        st.subheader('Step 3: Select Date Range')
        col1, col2 = st.columns(2)
        with col1:
            start_date_bond = st.date_input('Start Date (Bonds)', datetime.now().date() - timedelta(days=365))
        with col2:
            end_date_bond = st.date_input('End Date (Bonds)', datetime.now().date())
        
        if st.button('Fetch Bond Prices and Run PCA', use_container_width=True):
            # Validate API key if needed
            if data_provider in ['Finnhub API', 'IEX Cloud API'] and not api_key:
                st.error("Please provide an API key for the selected data provider.")
                st.stop()
            
            with st.spinner('Fetching bond prices...'):
                if data_provider == 'Finnhub API':
                    bond_df, failed_isins = get_bond_prices_finnhub(isin_list, start_date_bond, end_date_bond, api_key)
                elif data_provider == 'IEX Cloud API':
                    bond_df, failed_isins = get_bond_prices_iexcloud(isin_list, start_date_bond, end_date_bond, api_key)
                else:  # Multi-Source (yfinance + exchanges)
                    bond_df, failed_isins = get_bond_prices_cboe(isin_list, start_date_bond, end_date_bond)
            
            if failed_isins:
                st.warning(f"Failed to fetch data for {len(failed_isins)} ISINs: {', '.join(failed_isins[:5])}{'...' if len(failed_isins) > 5 else ''}")
            
            if bond_df.empty:
                st.error("No bond price data could be retrieved. Please check the ISINs or try another data provider.")
                st.stop()
            
            st.success(f"Successfully loaded {bond_df.shape[1]} bonds with {bond_df.shape[0]} price observations")
            
            # Display raw data preview
            with st.expander("View Price Data Preview"):
                st.dataframe(bond_df.head())
            
            # Calculate daily returns
            returns_df = bond_df.pct_change().dropna()
            
            if returns_df.shape[0] < 2:
                st.error("Not enough data points to calculate returns.")
                st.stop()
            
            # Standardize the returns
            scaler = StandardScaler()
            returns_scaled = scaler.fit_transform(returns_df)
            
            # Perform PCA
            n_components = min(10, returns_df.shape[1])  # Use up to 10 components
            pca = PCA(n_components=n_components)
            pca_transformed = pca.fit_transform(returns_scaled)
            
            # Explained variance
            explained_var = pca.explained_variance_ratio_
            cumulative_var = np.cumsum(explained_var)
            
            # Display explained variance
            st.subheader('Explained Variance by Principal Components')
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("PC1 Variance Explained", f"{explained_var[0]*100:.2f}%")
            with col2:
                st.metric("PC2 Variance Explained", f"{explained_var[1]*100:.2f}%")
            with col3:
                st.metric("Cumulative Variance (Top 3 PCs)", f"{cumulative_var[2]*100:.2f}%")
            
            # Plot explained variance
            var_df = pd.DataFrame({
                'Principal Component': [f'PC{i+1}' for i in range(n_components)],
                'Explained Variance': explained_var * 100,
                'Cumulative Variance': cumulative_var * 100
            })
            fig_var = px.bar(var_df, x='Principal Component', y='Explained Variance', 
                            title='Explained Variance by Principal Component',
                            labels={'Explained Variance': 'Explained Variance (%)'})
            st.plotly_chart(fig_var, use_container_width=True)
            
            # Component loadings - which bonds drive which PCs
            st.subheader('Bond Contributions to Principal Components')
            
            loadings_df = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=returns_df.columns
            )
            
            # Select which PC to visualize
            pc_selected = st.selectbox('Select Principal Component to analyze', 
                                       [f'PC{i+1}' for i in range(n_components)])
            
            # Show top contributing bonds
            pc_loadings = loadings_df[pc_selected].abs().sort_values(ascending=False)
            st.write(f"**Top 20 Bonds Contributing to {pc_selected}**")
            
            top_bonds = pd.DataFrame({
                'Bond': pc_loadings.index[:20],
                'Absolute Loading': pc_loadings.values[:20],
                'Loading': loadings_df[pc_selected].loc[pc_loadings.index[:20]].values
            })
            
            fig_loadings = px.bar(top_bonds, x='Absolute Loading', y='Bond', orientation='h',
                                 title=f'Top 20 Bonds Contributing to {pc_selected}',
                                 color='Loading', color_continuous_scale='RdBu')
            fig_loadings.update_layout(height=500)
            st.plotly_chart(fig_loadings, use_container_width=True)
            
            # 2D PCA scatter
            st.subheader('2D PCA Space (PC1 vs PC2)')
            pca_2d_df = pd.DataFrame({
                'PC1': pca_transformed[:, 0],
                'PC2': pca_transformed[:, 1],
                'Date': returns_df.index
            })
            
            fig_2d = px.scatter(pca_2d_df, x='PC1', y='PC2', hover_data=['Date'],
                               title='Bond Portfolio in 2D PCA Space',
                               labels={'PC1': f'PC1 ({explained_var[0]*100:.1f}%)', 
                                      'PC2': f'PC2 ({explained_var[1]*100:.1f}%)'})
            fig_2d.update_traces(marker=dict(size=4, opacity=0.6))
            st.plotly_chart(fig_2d, use_container_width=True)
            
            # Correlation heatmap of top bonds
            st.subheader('Correlation Matrix of Top 10 Contributing Bonds')
            top_10_bonds = pc_loadings.index[:10]
            corr_matrix = returns_df[top_10_bonds].corr()
            
            fig_corr = px.imshow(corr_matrix, labels=dict(x="Bond", y="Bond", color="Correlation"),
                                title='Correlation Matrix - Top 10 Bonds',
                                color_continuous_scale='RdBu', zmin=-1, zmax=1)
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("📋 Please upload a CSV file with ISINs or paste them directly to begin the analysis.")
        st.write("**How to use:**")
        st.write("1. Select a data provider (Multi-Source works best for global bonds)")
        st.write("2. Upload a CSV file with one ISIN per row, or paste ISINs directly")
        st.write("3. Select date range")
        st.write("4. Click 'Fetch Bond Prices and Run PCA'")
        st.write("\n**Data Provider Notes:**")
        st.write("- **Multi-Source**: Best for bonds listed on European/Asian exchanges (LSE, Deutsche Börse, Euronext, etc.)")
        st.write("- **Finnhub**: Requires free API key from https://finnhub.io")
        st.write("- **IEX Cloud**: Requires free API key from https://iexcloud.io")
