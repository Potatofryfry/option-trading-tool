import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import datetime
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Option Trading Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .signal-strong {
        color: #00C853;
        font-weight: bold;
    }
    .signal-moderate {
        color: #FFD600;
        font-weight: bold;
    }
    .signal-weak {
        color: #FF6D00;
        font-weight: bold;
    }
    .signal-none {
        color: #757575;
    }
    .put-signal-strong {
        color: #D50000;
        font-weight: bold;
    }
    .put-signal-moderate {
        color: #FF6D00;
        font-weight: bold;
    }
    .put-signal-weak {
        color: #FFD600;
        font-weight: bold;
    }
    .indicator-value {
        font-weight: bold;
        font-size: 1.2rem;
    }
    .recommendation-box {
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .call-box {
        background-color: rgba(0, 200, 83, 0.1);
        border: 1px solid #00C853;
    }
    .put-box {
        background-color: rgba(213, 0, 0, 0.1);
        border: 1px solid #D50000;
    }
    .neutral-box {
        background-color: rgba(117, 117, 117, 0.1);
        border: 1px solid #757575;
    }
</style>
""", unsafe_allow_html=True)

#####################################
# DATA FETCHER CLASS
#####################################

class DataFetcher:
    """
    Class for fetching stock data from Yahoo Finance API
    """
    def __init__(self):
        """Initialize the DataFetcher class"""
        pass
    
    def fetch_stock_data(self, symbol, interval='1d', range='6mo', region='US'):
        """
        Fetch stock data from Yahoo Finance API
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        interval : str
            Time interval between data points (1d, 1wk, 1mo)
        range : str
            Time range for data (1mo, 3mo, 6mo, 1y, 2y, 5y, max)
        region : str
            Region for the stock (US, BR, AU, CA, FR, DE, HK, IN, IT, ES, GB, SG)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with stock data
        """
        try:
            # For Streamlit Cloud deployment, we'll use yfinance instead of the API
            # This ensures the app works without any special API access
            import yfinance as yf
            
            # Fetch data using yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=range, interval=interval)
            
            # Rename columns to match our expected format
            data.columns = [col.lower() for col in data.columns]
            if 'adj close' in data.columns:
                data.rename(columns={'adj close': 'adjclose'}, inplace=True)
            
            # Add symbol as a column
            data['symbol'] = symbol
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            
            # For demonstration purposes, generate random data if API fails
            st.warning("Generating random data for demonstration purposes...")
            return self._generate_demo_data(symbol, range)
    
    def _generate_demo_data(self, symbol, range='6mo'):
        """Generate random data for demonstration purposes"""
        # Determine number of days based on range
        days_map = {
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365,
            '2y': 365*2,
            '5y': 365*5,
            'max': 365*10
        }
        days = days_map.get(range, 180)
        
        # Generate dates
        end_date = datetime.datetime.now()
        dates = [end_date - datetime.timedelta(days=i) for i in range(days, 0, -1)]
        
        # Set a random starting price between 50 and 200
        start_price = np.random.uniform(50, 200)
        
        # Generate random price movements with some trend and volatility
        price_changes = np.random.normal(0, 0.02, days)  # Daily returns with 2% volatility
        
        # Add a slight upward trend
        price_changes = price_changes + 0.001
        
        # Calculate prices
        prices = [start_price]
        for change in price_changes:
            prices.append(prices[-1] * (1 + change))
        prices = prices[1:]  # Remove the initial price
        
        # Generate OHLC data
        opens = prices
        closes = [price * (1 + np.random.normal(0, 0.005)) for price in prices]
        highs = [max(o, c) * (1 + abs(np.random.normal(0, 0.01))) for o, c in zip(opens, closes)]
        lows = [min(o, c) * (1 - abs(np.random.normal(0, 0.01))) for o, c in zip(opens, closes)]
        
        # Generate volume data
        volumes = [int(np.random.uniform(1000000, 10000000)) for _ in range(days)]
        
        # Create DataFrame
        data = {
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
            'adjclose': closes,  # Use close as adjusted close for simplicity
            'symbol': symbol
        }
        
        df = pd.DataFrame(data, index=dates)
        return df

#####################################
# TECHNICAL ANALYSIS CLASS
#####################################

class TechnicalAnalysis:
    """
    Class for calculating technical indicators for stock data
    """
    def __init__(self):
        """Initialize the TechnicalAnalysis class"""
        pass
    
    def calculate_all_indicators(self, data):
        """
        Calculate all technical indicators for the given data
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with stock data (must contain 'open', 'high', 'low', 'close', 'volume' columns)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with all technical indicators
        """
        if data is None or data.empty:
            return None
        
        # Create a copy of the data
        df = data.copy()
        
        # Calculate RSI
        df = self.calculate_rsi(df)
        
        # Calculate Bollinger Bands
        df = self.calculate_bollinger_bands(df)
        
        # Calculate MACD
        df = self.calculate_macd(df)
        
        # Calculate Stochastic Oscillator
        df = self.calculate_stochastic(df)
        
        # Calculate Moving Averages
        df = self.calculate_moving_averages(df)
        
        # Calculate Money Flow Index
        df = self.calculate_mfi(df)
        
        # Calculate Average True Range
        df = self.calculate_atr(df)
        
        # Calculate On-Balance Volume
        df = self.calculate_obv(df)
        
        # Calculate Intraday Momentum Index
        df = self.calculate_imi(df)
        
        # Calculate Rate of Change
        df = self.calculate_roc(df)
        
        return df
    
    def calculate_rsi(self, data, period=14):
        """
        Calculate Relative Strength Index (RSI)
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with stock data
        period : int
            Period for RSI calculation
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with RSI
        """
        df = data.copy()
        
        # Calculate price changes
        delta = df['close'].diff()
        
        # Separate gains and losses
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Add RSI to DataFrame
        df['RSI'] = rsi
        
        return df
    
    def calculate_bollinger_bands(self, data, period=20, std_dev=2):
        """
        Calculate Bollinger Bands
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with stock data
        period : int
            Period for moving average
        std_dev : int
            Number of standard deviations for bands
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with Bollinger Bands
        """
        df = data.copy()
        
        # Calculate middle band (SMA)
        df['BB_Middle'] = df['close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        df['BB_Std'] = df['close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * std_dev)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * std_dev)
        
        # Calculate bandwidth
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Calculate %B
        df['BB_B'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        return df
    
    def calculate_macd(self, data, fast_period=12, slow_period=26, signal_period=9):
        """
        Calculate Moving Average Convergence Divergence (MACD)
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with stock data
        fast_period : int
            Period for fast EMA
        slow_period : int
            Period for slow EMA
        signal_period : int
            Period for signal line
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with MACD
        """
        df = data.copy()
        
        # Calculate fast and slow EMAs
        df['EMA_Fast'] = df['close'].ewm(span=fast_period, adjust=False).mean()
        df['EMA_Slow'] = df['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        df['MACD'] = df['EMA_Fast'] - df['EMA_Slow']
        
        # Calculate signal line
        df['MACD_Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        return df
    
    def calculate_stochastic(self, data, k_period=14, d_period=3):
        """
        Calculate Stochastic Oscillator
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with stock data
        k_period : int
            Period for %K
        d_period : int
            Period for %D
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with Stochastic Oscillator
        """
        df = data.copy()
        
        # Calculate %K
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        df['Stoch_K'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
        
        # Calculate %D
        df['Stoch_D'] = df['Stoch_K'].rolling(window=d_period).mean()
        
        return df
    
    def calculate_moving_averages(self, data, periods=[10, 20, 50, 200]):
        """
        Calculate Simple Moving Averages (SMA)
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with stock data
        periods : list
            List of periods for SMAs
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with SMAs
        """
        df = data.copy()
        
        # Calculate SMAs for each period
        for period in periods:
            df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
        
        return df
    
    def calculate_mfi(self, data, period=14):
        """
        Calculate Money Flow Index (MFI)
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with stock data
        period : int
            Period for MFI calculation
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with MFI
        """
        df = data.copy()
        
        # Calculate typical price
        df['TP'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate raw money flow
        df['MF'] = df['TP'] * df['volume']
        
        # Calculate price change
        df['Price_Change'] = df['TP'].diff()
        
        # Separate positive and negative money flow
        df['Positive_MF'] = 0
        df['Negative_MF'] = 0
        
        # Use numpy arrays for better performance
        price_change = df['Price_Change'].values
        raw_money_flow = df['MF'].values
        money_flow_positive = df['Positive_MF'].values
        money_flow_negative = df['Negative_MF'].values
        
        # Assign values based on price change
        for i in range(len(price_change)):
            if price_change[i] > 0:
                money_flow_positive[i] = raw_money_flow[i]
            elif price_change[i] < 0:
                money_flow_negative[i] = raw_money_flow[i]
        
        # Update DataFrame
        df['Positive_MF'] = money_flow_positive
        df['Negative_MF'] = money_flow_negative
        
        # Calculate money flow ratio
        df['Positive_MF_Sum'] = df['Positive_MF'].rolling(window=period).sum()
        df['Negative_MF_Sum'] = df['Negative_MF'].rolling(window=period).sum()
        df['MF_Ratio'] = df['Positive_MF_Sum'] / df['Negative_MF_Sum']
        
        # Calculate MFI
        df['MFI'] = 100 - (100 / (1 + df['MF_Ratio']))
        
        return df
    
    def calculate_atr(self, data, period=14):
        """
        Calculate Average True Range (ATR)
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with stock data
        period : int
            Period for ATR calculation
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with ATR
        """
        df = data.copy()
        
        # Calculate true range
        df['TR1'] = abs(df['high'] - df['low'])
        df['TR2'] = abs(df['high'] - df['close'].shift())
        df['TR3'] = abs(df['low'] - df['close'].shift())
        df['TR'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
        
        # Calculate ATR
        df['ATR'] = df['TR'].rolling(window=period).mean()
        
        # Drop temporary columns
        df = df.drop(['TR1', 'TR2', 'TR3', 'TR'], axis=1)
        
        return df
    
    def calculate_obv(self, data):
        """
        Calculate On-Balance Volume (OBV)
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with stock data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with OBV
        """
        df = data.copy()
        
        # Calculate price change
        df['Price_Change'] = df['close'].diff()
        
        # Calculate OBV
        df['OBV'] = 0
        df.loc[df['Price_Change'] > 0, 'OBV'] = df['volume']
        df.loc[df['Price_Change'] < 0, 'OBV'] = -df['volume']
        df.loc[df['Price_Change'] == 0, 'OBV'] = 0
        df['OBV'] = df['OBV'].cumsum()
        
        # Drop temporary columns
        df = df.drop(['Price_Change'], axis=1)
        
        return df
    
    def calculate_imi(self, data, period=14):
        """
        Calculate Intraday Momentum Index (IMI)
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with stock data
        period : int
            Period for IMI calculation
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with IMI
        """
        df = data.copy()
        
        # Calculate daily change
        df['Daily_Change'] = df['close'] - df['open']
        
        # Separate up and down days
        df['Up_Days'] = 0
        df['Down_Days'] = 0
        
        # Use numpy arrays for better performance
        daily_change = df['Daily_Change'].values
        up_days = df['Up_Days'].values
        down_days = df['Down_Days'].values
        
        # Assign values based on daily change
        for i in range(len(daily_change)):
            if daily_change[i] > 0:
                up_days[i] = daily_change[i]
            elif daily_change[i] < 0:
                down_days[i] = -daily_change[i]
        
        # Update DataFrame
        df['Up_Days'] = up_days
        df['Down_Days'] = down_days
        
        # Calculate sums
        df['Up_Sum'] = df['Up_Days'].rolling(window=period).sum()
        df['Down_Sum'] = df['Down_Days'].rolling(window=period).sum()
        
        # Calculate IMI
        df['IMI'] = 100 * (df['Up_Sum'] / (df['Up_Sum'] + df['Down_Sum']))
        
        # Drop temporary columns
        df = df.drop(['Daily_Change', 'Up_Days', 'Down_Days', 'Up_Sum', 'Down_Sum'], axis=1)
        
        return df
    
    def calculate_roc(self, data, period=12):
        """
        Calculate Rate of Change (ROC)
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with stock data
        period : int
            Period for ROC calculation
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with ROC
        """
        df = data.copy()
        
        # Calculate ROC
        df['ROC'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
        
        return df

#####################################
# SIGNAL GENERATOR CLASS
#####################################

class SignalGenerator:
    """
    Class for generating trading signals based on technical indicators
    """
    def __init__(self):
        """Initialize the SignalGenerator class"""
        pass
    
    def generate_combined_signals(self, indicators):
        """
        Generate combined trading signals based on multiple technical indicators
        
        Parameters:
        -----------
        indicators : pandas.DataFrame
            DataFrame with technical indicators
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with combined signals
        """
        if indicators is None or indicators.empty:
            return None
        
        # Create a copy of the indicators
        signals = indicators.copy()
        
        # Initialize signal columns
        signals['RSI_Call'] = 0
        signals['RSI_Put'] = 0
        signals['RSI_Call_Strength'] = 0
        signals['RSI_Put_Strength'] = 0
        
        signals['BB_Call'] = 0
        signals['BB_Put'] = 0
        signals['BB_Call_Strength'] = 0
        signals['BB_Put_Strength'] = 0
        
        signals['MACD_Call'] = 0
        signals['MACD_Put'] = 0
        signals['MACD_Call_Strength'] = 0
        signals['MACD_Put_Strength'] = 0
        
        signals['Stoch_Call'] = 0
        signals['Stoch_Put'] = 0
        signals['Stoch_Call_Strength'] = 0
        signals['Stoch_Put_Strength'] = 0
        
        signals['MA_Call'] = 0
        signals['MA_Put'] = 0
        signals['MA_Call_Strength'] = 0
        signals['MA_Put_Strength'] = 0
        
        signals['MFI_Call'] = 0
        signals['MFI_Put'] = 0
        signals['MFI_Call_Strength'] = 0
        signals['MFI_Put_Strength'] = 0
        
        signals['Vol_Call'] = 0
        signals['Vol_Put'] = 0
        signals['Vol_Call_Strength'] = 0
        signals['Vol_Put_Strength'] = 0
        
        # Generate signals for each indicator
        signals = self._generate_rsi_signals(signals)
        signals = self._generate_bollinger_signals(signals)
        signals = self._generate_macd_signals(signals)
        signals = self._generate_stochastic_signals(signals)
        signals = self._generate_moving_average_signals(signals)
        signals = self._generate_mfi_signals(signals)
        signals = self._generate_volume_signals(signals)
        
        # Generate combined signals
        signals = self._generate_combined_signal_scores(signals)
        
        return signals
    
    def _generate_rsi_signals(self, signals):
        """Generate signals based on RSI"""
        if 'RSI' not in signals.columns:
            return signals
        
        # Calculate RSI changes
        signals['RSI_Change'] = signals['RSI'].diff()
        
        # Create masks for different conditions
        oversold = signals['RSI'] < 30
        overbought = signals['RSI'] > 70
        
        rising_rsi = signals['RSI_Change'] > 0
        falling_rsi = signals['RSI_Change'] < 0
        
        # Generate call signals
        signals.loc[oversold & rising_rsi, 'RSI_Call'] = 1
        signals.loc[oversold & rising_rsi, 'RSI_Call_Strength'] = 1.0
        
        signals.loc[(signals['RSI'] > 30) & (signals['RSI'] < 40) & rising_rsi, 'RSI_Call'] = 1
        signals.loc[(signals['RSI'] > 30) & (signals['RSI'] < 40) & rising_rsi, 'RSI_Call_Strength'] = 0.7
        
        # Generate put signals
        signals.loc[overbought & falling_rsi, 'RSI_Put'] = 1
        signals.loc[overbought & falling_rsi, 'RSI_Put_Strength'] = 1.0
        
        signals.loc[(signals['RSI'] > 60) & (signals['RSI'] < 70) & falling_rsi, 'RSI_Put'] = 1
        signals.loc[(signals['RSI'] > 60) & (signals['RSI'] < 70) & falling_rsi, 'RSI_Put_Strength'] = 0.7
        
        return signals
    
    def _generate_bollinger_signals(self, signals):
        """Generate signals based on Bollinger Bands"""
        if 'BB_Lower' not in signals.columns or 'BB_Upper' not in signals.columns:
            return signals
        
        # Calculate price changes
        signals['Price_Change'] = signals['close'].diff()
        
        # Create masks for different conditions
        lower_band_touch = signals['close'] <= signals['BB_Lower']
        upper_band_touch = signals['close'] >= signals['BB_Upper']
        
        lower_middle_range = (signals['close'] > signals['BB_Lower']) & (signals['close'] < signals['BB_Middle'])
        upper_middle_range = (signals['close'] < signals['BB_Upper']) & (signals['close'] > signals['BB_Middle'])
        
        middle_band_range = (signals['close'] > signals['BB_Middle'] * 0.98) & (signals['close'] < signals['BB_Middle'] * 1.02)
        
        rising_price = signals['Price_Change'] > 0
        falling_price = signals['Price_Change'] < 0
        
        # Generate call signals
        signals.loc[lower_band_touch & rising_price, 'BB_Call'] = 1
        signals.loc[lower_band_touch & rising_price, 'BB_Call_Strength'] = 1.0
        
        signals.loc[lower_middle_range & rising_price, 'BB_Call'] = 1
        signals.loc[lower_middle_range & rising_price, 'BB_Call_Strength'] = 0.8
        
        signals.loc[middle_band_range & rising_price, 'BB_Call'] = 1
        signals.loc[middle_band_range & rising_price, 'BB_Call_Strength'] = 0.7
        
        # Generate put signals
        signals.loc[upper_band_touch & falling_price, 'BB_Put'] = 1
        signals.loc[upper_band_touch & falling_price, 'BB_Put_Strength'] = 1.0
        
        signals.loc[upper_middle_range & falling_price, 'BB_Put'] = 1
        signals.loc[upper_middle_range & falling_price, 'BB_Put_Strength'] = 0.7
        
        return signals
    
    def _generate_macd_signals(self, signals):
        """Generate signals based on MACD"""
        if 'MACD' not in signals.columns or 'MACD_Signal' not in signals.columns:
            return signals
        
        # Calculate MACD and histogram changes
        signals['MACD_Change'] = signals['MACD'].diff()
        signals['MACD_Signal_Change'] = signals['MACD_Signal'].diff()
        signals['MACD_Hist_Change'] = signals['MACD_Hist'].diff()
        
        # Create masks for different conditions
        macd_above_signal = signals['MACD'] > signals['MACD_Signal']
        macd_below_signal = signals['MACD'] < signals['MACD_Signal']
        
        macd_crossing_above = (signals['MACD'].shift() < signals['MACD_Signal'].shift()) & macd_above_signal
        macd_crossing_below = (signals['MACD'].shift() > signals['MACD_Signal'].shift()) & macd_below_signal
        
        positive_hist = signals['MACD_Hist'] > 0
        negative_hist = signals['MACD_Hist'] < 0
        
        increasing_hist = signals['MACD_Hist_Change'] > 0
        decreasing_hist = signals['MACD_Hist_Change'] < 0
        
        # Generate call signals
        signals.loc[macd_crossing_above, 'MACD_Call'] = 1
        signals.loc[macd_crossing_above, 'MACD_Call_Strength'] = 1.0
        
        signals.loc[positive_hist & increasing_hist, 'MACD_Call'] = 1
        signals.loc[positive_hist & increasing_hist, 'MACD_Call_Strength'] = 0.8
        
        signals.loc[negative_hist & increasing_hist, 'MACD_Call'] = 1
        signals.loc[negative_hist & increasing_hist, 'MACD_Call_Strength'] = 0.7
        
        # Generate put signals
        signals.loc[macd_crossing_below, 'MACD_Put'] = 1
        signals.loc[macd_crossing_below, 'MACD_Put_Strength'] = 1.0
        
        signals.loc[negative_hist & decreasing_hist, 'MACD_Put'] = 1
        signals.loc[negative_hist & decreasing_hist, 'MACD_Put_Strength'] = 0.8
        
        signals.loc[positive_hist & decreasing_hist, 'MACD_Put'] = 1
        signals.loc[positive_hist & decreasing_hist, 'MACD_Put_Strength'] = 0.7
        
        return signals
    
    def _generate_stochastic_signals(self, signals):
        """Generate signals based on Stochastic Oscillator"""
        if 'Stoch_K' not in signals.columns or 'Stoch_D' not in signals.columns:
            return signals
        
        # Calculate Stochastic changes
        signals['Stoch_K_Change'] = signals['Stoch_K'].diff()
        signals['Stoch_D_Change'] = signals['Stoch_D'].diff()
        
        # Create masks for different conditions
        oversold = signals['Stoch_K'] < 20
        overbought = signals['Stoch_K'] > 80
        
        k_rising = signals['Stoch_K_Change'] > 0
        k_falling = signals['Stoch_K_Change'] < 0
        
        k_above_d = signals['Stoch_K'] > signals['Stoch_D']
        k_below_d = signals['Stoch_K'] < signals['Stoch_D']
        
        k_crossing_above_d = (signals['Stoch_K'].shift() < signals['Stoch_D'].shift()) & k_above_d
        k_crossing_below_d = (signals['Stoch_K'].shift() > signals['Stoch_D'].shift()) & k_below_d
        
        # Generate call signals
        signals.loc[oversold & k_crossing_above_d, 'Stoch_Call'] = 1
        signals.loc[oversold & k_crossing_above_d, 'Stoch_Call_Strength'] = 1.0
        
        signals.loc[oversold & k_rising & k_below_d, 'Stoch_Call'] = 1
        signals.loc[oversold & k_rising & k_below_d, 'Stoch_Call_Strength'] = 0.7
        
        signals.loc[(signals['Stoch_K'] > 20) & (signals['Stoch_K'] < 40) & k_crossing_above_d, 'Stoch_Call'] = 1
        signals.loc[(signals['Stoch_K'] > 20) & (signals['Stoch_K'] < 40) & k_crossing_above_d, 'Stoch_Call_Strength'] = 0.8
        
        # Generate put signals
        signals.loc[overbought & k_crossing_below_d, 'Stoch_Put'] = 1
        signals.loc[overbought & k_crossing_below_d, 'Stoch_Put_Strength'] = 1.0
        
        signals.loc[overbought & k_falling & k_above_d, 'Stoch_Put'] = 1
        signals.loc[overbought & k_falling & k_above_d, 'Stoch_Put_Strength'] = 0.7
        
        signals.loc[(signals['Stoch_K'] > 60) & (signals['Stoch_K'] < 80) & k_crossing_below_d, 'Stoch_Put'] = 1
        signals.loc[(signals['Stoch_K'] > 60) & (signals['Stoch_K'] < 80) & k_crossing_below_d, 'Stoch_Put_Strength'] = 0.8
        
        return signals
    
    def _generate_moving_average_signals(self, signals):
        """Generate signals based on Moving Averages"""
        if 'SMA_50' not in signals.columns or 'SMA_200' not in signals.columns:
            return signals
        
        # Calculate price changes
        signals['Price_Change'] = signals['close'].diff()
        
        # Create masks for different conditions
        price_above_sma50 = signals['close'] > signals['SMA_50']
        price_below_sma50 = signals['close'] < signals['SMA_50']
        
        price_above_sma200 = signals['close'] > signals['SMA_200']
        price_below_sma200 = signals['close'] < signals['SMA_200']
        
        sma50_above_sma200 = signals['SMA_50'] > signals['SMA_200']
        sma50_below_sma200 = signals['SMA_50'] < signals['SMA_200']
        
        golden_cross = (signals['SMA_50'].shift() < signals['SMA_200'].shift()) & sma50_above_sma200
        death_cross = (signals['SMA_50'].shift() > signals['SMA_200'].shift()) & sma50_below_sma200
        
        approaching_sma = (signals['close'] > signals['SMA_50'] * 0.98) & (signals['close'] < signals['SMA_50'] * 1.02)
        approaching_sma_above = (signals['close'] < signals['SMA_50'] * 1.05) & (signals['close'] > signals['SMA_50'])
        
        rising_price = signals['Price_Change'] > 0
        falling_price = signals['Price_Change'] < 0
        
        # Generate call signals
        signals.loc[golden_cross, 'MA_Call'] = 1
        signals.loc[golden_cross, 'MA_Call_Strength'] = 1.0
        
        signals.loc[price_above_sma50 & price_above_sma200 & sma50_above_sma200 & rising_price, 'MA_Call'] = 1
        signals.loc[price_above_sma50 & price_above_sma200 & sma50_above_sma200 & rising_price, 'MA_Call_Strength'] = 0.9
        
        signals.loc[approaching_sma & rising_price, 'MA_Call'] = 1
        signals.loc[approaching_sma & rising_price, 'MA_Call_Strength'] = 0.7
        
        signals.loc[price_below_sma50 & price_above_sma200 & rising_price, 'MA_Call'] = 1
        signals.loc[price_below_sma50 & price_above_sma200 & rising_price, 'MA_Call_Strength'] = 0.6
        
        # Generate put signals
        signals.loc[death_cross, 'MA_Put'] = 1
        signals.loc[death_cross, 'MA_Put_Strength'] = 1.0
        
        signals.loc[price_below_sma50 & price_below_sma200 & sma50_below_sma200 & falling_price, 'MA_Put'] = 1
        signals.loc[price_below_sma50 & price_below_sma200 & sma50_below_sma200 & falling_price, 'MA_Put_Strength'] = 0.9
        
        signals.loc[approaching_sma_above & falling_price, 'MA_Put'] = 1
        signals.loc[approaching_sma_above & falling_price, 'MA_Put_Strength'] = 0.7
        
        signals.loc[price_above_sma50 & price_below_sma200 & falling_price, 'MA_Put'] = 1
        signals.loc[price_above_sma50 & price_below_sma200 & falling_price, 'MA_Put_Strength'] = 0.6
        
        return signals
    
    def _generate_mfi_signals(self, signals):
        """Generate signals based on Money Flow Index"""
        if 'MFI' not in signals.columns:
            return signals
        
        # Calculate MFI changes
        signals['MFI_Change'] = signals['MFI'].diff()
        
        # Create masks for different conditions
        low_mfi = signals['MFI'] < 20
        high_mfi = signals['MFI'] > 80
        
        rising_mfi = signals['MFI_Change'] > 0
        falling_mfi = signals['MFI_Change'] < 0
        
        # Generate call signals
        signals.loc[low_mfi & rising_mfi, 'MFI_Call'] = 1
        signals.loc[low_mfi & rising_mfi, 'MFI_Call_Strength'] = 0.7
        
        signals.loc[(signals['MFI'] > 20) & (signals['MFI'] < 30) & rising_mfi, 'MFI_Call'] = 1
        signals.loc[(signals['MFI'] > 20) & (signals['MFI'] < 30) & rising_mfi, 'MFI_Call_Strength'] = 0.6
        
        # Generate put signals
        signals.loc[high_mfi & falling_mfi, 'MFI_Put'] = 1
        signals.loc[high_mfi & falling_mfi, 'MFI_Put_Strength'] = 0.7
        
        signals.loc[(signals['MFI'] > 70) & (signals['MFI'] < 80) & falling_mfi, 'MFI_Put'] = 1
        signals.loc[(signals['MFI'] > 70) & (signals['MFI'] < 80) & falling_mfi, 'MFI_Put_Strength'] = 0.6
        
        return signals
    
    def _generate_volume_signals(self, signals):
        """Generate signals based on Volume"""
        if 'volume' not in signals.columns:
            return signals
        
        # Calculate volume changes
        signals['Volume_Change'] = signals['volume'].diff()
        signals['Volume_Ratio'] = signals['volume'] / signals['volume'].rolling(window=20).mean()
        
        # Calculate price changes
        signals['Price_Change'] = signals['close'].diff()
        
        # Create masks for different conditions
        high_volume = signals['Volume_Ratio'] > 1.5
        moderate_volume = (signals['Volume_Ratio'] > 1.2) & (signals['Volume_Ratio'] <= 1.5)
        
        increasing_volume = signals['Volume_Change'] > 0
        
        price_up = signals['Price_Change'] > 0
        price_down = signals['Price_Change'] < 0
        
        # Generate call signals
        signals.loc[price_up & high_volume & increasing_volume, 'Vol_Call'] = 1
        signals.loc[price_up & high_volume & increasing_volume, 'Vol_Call_Strength'] = 0.8
        
        signals.loc[price_up & moderate_volume, 'Vol_Call'] = 1
        signals.loc[price_up & moderate_volume, 'Vol_Call_Strength'] = 0.7
        
        # Generate put signals
        signals.loc[price_down & high_volume & increasing_volume, 'Vol_Put'] = 1
        signals.loc[price_down & high_volume & increasing_volume, 'Vol_Put_Strength'] = 0.8
        
        signals.loc[price_down & moderate_volume, 'Vol_Put'] = 1
        signals.loc[price_down & moderate_volume, 'Vol_Put_Strength'] = 0.7
        
        return signals
    
    def _generate_combined_signal_scores(self, signals):
        """Generate combined signal scores"""
        # Define weights for each indicator
        weights = {
            'RSI': 0.15,
            'BB': 0.15,
            'MACD': 0.20,
            'Stoch': 0.15,
            'MA': 0.20,
            'MFI': 0.10,
            'Vol': 0.05
        }
        
        # Calculate weighted call score
        signals['Call_Score'] = (
            weights['RSI'] * signals['RSI_Call'] * signals['RSI_Call_Strength'] +
            weights['BB'] * signals['BB_Call'] * signals['BB_Call_Strength'] +
            weights['MACD'] * signals['MACD_Call'] * signals['MACD_Call_Strength'] +
            weights['Stoch'] * signals['Stoch_Call'] * signals['Stoch_Call_Strength'] +
            weights['MA'] * signals['MA_Call'] * signals['MA_Call_Strength'] +
            weights['MFI'] * signals['MFI_Call'] * signals['MFI_Call_Strength'] +
            weights['Vol'] * signals['Vol_Call'] * signals['Vol_Call_Strength']
        ) * 100
        
        # Calculate weighted put score
        signals['Put_Score'] = (
            weights['RSI'] * signals['RSI_Put'] * signals['RSI_Put_Strength'] +
            weights['BB'] * signals['BB_Put'] * signals['BB_Put_Strength'] +
            weights['MACD'] * signals['MACD_Put'] * signals['MACD_Put_Strength'] +
            weights['Stoch'] * signals['Stoch_Put'] * signals['Stoch_Put_Strength'] +
            weights['MA'] * signals['MA_Put'] * signals['MA_Put_Strength'] +
            weights['MFI'] * signals['MFI_Put'] * signals['MFI_Put_Strength'] +
            weights['Vol'] * signals['Vol_Put'] * signals['Vol_Put_Strength']
        ) * 100
        
        return signals
    
    def generate_option_recommendations(self, indicators, signals):
        """
        Generate option recommendations based on signals
        
        Parameters:
        -----------
        indicators : pandas.DataFrame
            DataFrame with technical indicators
        signals : pandas.DataFrame
            DataFrame with trading signals
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with option recommendations
        """
        if signals is None or signals.empty:
            return None
        
        # Create a copy of the signals
        recommendations = signals.copy()
        
        # Keep only necessary columns
        recommendations = recommendations[['close', 'Call_Score', 'Put_Score']]
        
        # Add recommendation columns
        recommendations['Call_Strength'] = 'None'
        recommendations['Put_Strength'] = 'None'
        recommendations['Call_Recommendation'] = 'No Action'
        recommendations['Put_Recommendation'] = 'No Action'
        recommendations['Option_Expiration'] = 'N/A'
        recommendations['Strike_Selection'] = 'N/A'
        
        # Determine call strength
        recommendations.loc[recommendations['Call_Score'] >= 80, 'Call_Strength'] = 'Strong'
        recommendations.loc[(recommendations['Call_Score'] >= 60) & (recommendations['Call_Score'] < 80), 'Call_Strength'] = 'Moderate'
        recommendations.loc[(recommendations['Call_Score'] >= 40) & (recommendations['Call_Score'] < 60), 'Call_Strength'] = 'Weak'
        
        # Determine put strength
        recommendations.loc[recommendations['Put_Score'] >= 80, 'Put_Strength'] = 'Strong'
        recommendations.loc[(recommendations['Put_Score'] >= 60) & (recommendations['Put_Score'] < 80), 'Put_Strength'] = 'Moderate'
        recommendations.loc[(recommendations['Put_Score'] >= 40) & (recommendations['Put_Score'] < 60), 'Put_Strength'] = 'Weak'
        
        # Generate call recommendations
        recommendations.loc[recommendations['Call_Strength'] == 'Strong', 'Call_Recommendation'] = 'Buy Call Options'
        recommendations.loc[recommendations['Call_Strength'] == 'Moderate', 'Call_Recommendation'] = 'Consider Call Options'
        recommendations.loc[recommendations['Call_Strength'] == 'Weak', 'Call_Recommendation'] = 'Monitor for Call Opportunity'
        
        # Generate put recommendations
        recommendations.loc[recommendations['Put_Strength'] == 'Strong', 'Put_Recommendation'] = 'Buy Put Options'
        recommendations.loc[recommendations['Put_Strength'] == 'Moderate', 'Put_Recommendation'] = 'Consider Put Options'
        recommendations.loc[recommendations['Put_Strength'] == 'Weak', 'Put_Recommendation'] = 'Monitor for Put Opportunity'
        
        # Generate option expiration recommendations
        conditions = [
            (recommendations['Call_Strength'] == 'Strong') | (recommendations['Put_Strength'] == 'Strong'),
            (recommendations['Call_Strength'] == 'Moderate') | (recommendations['Put_Strength'] == 'Moderate'),
            (recommendations['Call_Strength'] == 'Weak') | (recommendations['Put_Strength'] == 'Weak')
        ]
        
        choices = [
            '2-4 weeks',
            '1-2 months',
            '2-3 months'
        ]
        
        recommendations['Option_Expiration'] = np.select(conditions, choices, default='N/A')
        
        # Generate strike price recommendations
        call_conditions = [
            recommendations['Call_Strength'] == 'Strong',
            recommendations['Call_Strength'] == 'Moderate',
            recommendations['Call_Strength'] == 'Weak'
        ]
        
        put_conditions = [
            recommendations['Put_Strength'] == 'Strong',
            recommendations['Put_Strength'] == 'Moderate',
            recommendations['Put_Strength'] == 'Weak'
        ]
        
        call_choices = [
            f'ATM to 5% OTM Call (Strike: ${recommendations["close"] * 1.025:.2f}-${recommendations["close"] * 1.05:.2f})',
            f'ATM to 3% OTM Call (Strike: ${recommendations["close"] * 1.015:.2f}-${recommendations["close"] * 1.03:.2f})',
            f'ATM Call (Strike: ${recommendations["close"]:.2f})'
        ]
        
        put_choices = [
            f'ATM to 5% OTM Put (Strike: ${recommendations["close"] * 0.95:.2f}-${recommendations["close"] * 0.975:.2f})',
            f'ATM to 3% OTM Put (Strike: ${recommendations["close"] * 0.97:.2f}-${recommendations["close"] * 0.985:.2f})',
            f'ATM Put (Strike: ${recommendations["close"]:.2f})'
        ]
        
        # Apply strike price recommendations
        call_strikes = np.select(call_conditions, call_choices, default='N/A')
        put_strikes = np.select(put_conditions, put_choices, default='N/A')
        
        # Combine strike recommendations based on which signal is stronger
        recommendations['Strike_Selection'] = np.where(
            recommendations['Call_Score'] > recommendations['Put_Score'],
            call_strikes,
            put_strikes
        )
        
        # Set to N/A if both signals are weak
        recommendations.loc[
            (recommendations['Call_Strength'] == 'None') & 
            (recommendations['Put_Strength'] == 'None'),
            'Strike_Selection'
        ] = 'N/A'
        
        # Add Close price for reference
        recommendations['Close'] = recommendations['close']
        
        return recommendations
    
    def generate_summary_report(self, symbol, indicators, recommendations):
        """
        Generate a summary report of the analysis
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        indicators : pandas.DataFrame
            DataFrame with technical indicators
        recommendations : pandas.DataFrame
            DataFrame with option recommendations
            
        Returns:
        --------
        str
            Report text
        """
        if indicators is None or indicators.empty or recommendations is None or recommendations.empty:
            return None
        
        # Get the latest data
        latest_date = recommendations.index[-1]
        latest_rec = recommendations.iloc[-1]
        latest_indicators = indicators.loc[latest_date]
        
        # Create the report
        report = f"""
        # Option Trading Analysis Report for {symbol}
        
        **Date:** {latest_date.strftime('%Y-%m-%d')}
        **Current Price:** ${latest_indicators['close']:.2f}
        
        ## Signal Summary
        
        ### Call Option Signal
        - **Score:** {latest_rec['Call_Score']:.1f}/100
        - **Strength:** {latest_rec['Call_Strength']}
        - **Recommendation:** {latest_rec['Call_Recommendation']}
        
        ### Put Option Signal
        - **Score:** {latest_rec['Put_Score']:.1f}/100
        - **Strength:** {latest_rec['Put_Strength']}
        - **Recommendation:** {latest_rec['Put_Recommendation']}
        
        ## Option Strategy
        - **Recommended Expiration:** {latest_rec['Option_Expiration']}
        - **Strike Price Selection:** {latest_rec['Strike_Selection']}
        
        ## Technical Indicators
        
        ### Momentum Indicators
        - **RSI:** {latest_indicators['RSI']:.1f}
        - **Stochastic %K:** {latest_indicators['Stoch_K']:.1f}
        - **Stochastic %D:** {latest_indicators['Stoch_D']:.1f}
        - **MACD:** {latest_indicators['MACD']:.3f}
        - **MACD Signal:** {latest_indicators['MACD_Signal']:.3f}
        - **MACD Histogram:** {latest_indicators['MACD_Hist']:.3f}
        
        ### Trend Indicators
        - **50-day SMA:** {latest_indicators['SMA_50']:.2f}
        - **200-day SMA:** {latest_indicators['SMA_200']:.2f}
        
        ### Volatility Indicators
        - **Bollinger Upper Band:** {latest_indicators['BB_Upper']:.2f}
        - **Bollinger Middle Band:** {latest_indicators['BB_Middle']:.2f}
        - **Bollinger Lower Band:** {latest_indicators['BB_Lower']:.2f}
        
        ### Volume Indicators
        - **Money Flow Index:** {latest_indicators['MFI']:.1f}
        
        ## Analysis Details
        
        """
        
        # Add indicator-specific analysis
        
        # RSI Analysis
        rsi_value = latest_indicators['RSI']
        if rsi_value < 30:
            report += "**RSI Analysis:** The RSI is currently in oversold territory, suggesting a potential buying opportunity for call options.\n\n"
        elif rsi_value > 70:
            report += "**RSI Analysis:** The RSI is currently in overbought territory, suggesting a potential buying opportunity for put options.\n\n"
        else:
            report += f"**RSI Analysis:** The RSI is currently at {rsi_value:.1f}, which is in neutral territory.\n\n"
        
        # MACD Analysis
        macd = latest_indicators['MACD']
        macd_signal = latest_indicators['MACD_Signal']
        if macd > macd_signal:
            report += "**MACD Analysis:** The MACD is above the signal line, indicating bullish momentum.\n\n"
        else:
            report += "**MACD Analysis:** The MACD is below the signal line, indicating bearish momentum.\n\n"
        
        # Bollinger Bands Analysis
        bb_upper = latest_indicators['BB_Upper']
        bb_lower = latest_indicators['BB_Lower']
        bb_middle = latest_indicators['BB_Middle']
        close = latest_indicators['close']
        
        if close > bb_upper:
            report += "**Bollinger Bands Analysis:** The price is above the upper Bollinger Band, suggesting overbought conditions.\n\n"
        elif close < bb_lower:
            report += "**Bollinger Bands Analysis:** The price is below the lower Bollinger Band, suggesting oversold conditions.\n\n"
        else:
            report += "**Bollinger Bands Analysis:** The price is within the Bollinger Bands, suggesting normal trading conditions.\n\n"
        
        # Moving Average Analysis
        sma_50 = latest_indicators['SMA_50']
        sma_200 = latest_indicators['SMA_200']
        
        if sma_50 > sma_200:
            report += "**Moving Average Analysis:** The 50-day SMA is above the 200-day SMA, indicating a bullish trend.\n\n"
        else:
            report += "**Moving Average Analysis:** The 50-day SMA is below the 200-day SMA, indicating a bearish trend.\n\n"
        
        return report
    
    def plot_signals(self, indicators, signals, symbol):
        """
        Plot the signals on a price chart
        
        Parameters:
        -----------
        indicators : pandas.DataFrame
            DataFrame with technical indicators
        signals : pandas.DataFrame
            DataFrame with trading signals
        symbol : str
            Stock symbol
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if indicators is None or indicators.empty or signals is None or signals.empty:
            return None
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot the price
        ax.plot(indicators.index, indicators['close'], label='Close Price', color='blue')
        
        # Plot the Bollinger Bands
        ax.plot(indicators.index, indicators['BB_Upper'], label='Upper BB', color='gray', linestyle='--')
        ax.plot(indicators.index, indicators['BB_Middle'], label='Middle BB', color='gray')
        ax.plot(indicators.index, indicators['BB_Lower'], label='Lower BB', color='gray', linestyle='--')
        
        # Plot the moving averages
        ax.plot(indicators.index, indicators['SMA_50'], label='50-day SMA', color='orange')
        ax.plot(indicators.index, indicators['SMA_200'], label='200-day SMA', color='red')
        
        # Plot call signals
        call_signals = signals[signals['Call_Score'] >= 80]
        ax.scatter(call_signals.index, call_signals['close'], marker='^', color='green', s=100, label='Strong Call Signal')
        
        # Plot put signals
        put_signals = signals[signals['Put_Score'] >= 80]
        ax.scatter(put_signals.index, put_signals['close'], marker='v', color='red', s=100, label='Strong Put Signal')
        
        # Add title and labels
        ax.set_title(f'{symbol} Price Chart with Option Signals')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        
        return fig

#####################################
# MAIN APPLICATION
#####################################

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'indicators' not in st.session_state:
    st.session_state.indicators = None
if 'signals' not in st.session_state:
    st.session_state.signals = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'symbol' not in st.session_state:
    st.session_state.symbol = None
if 'report' not in st.session_state:
    st.session_state.report = None

# Header
st.markdown("<h1 class='main-header'>Option Trading Signal Generator</h1>", unsafe_allow_html=True)
st.markdown("A tool to help determine optimal entry points for call or put options based on technical analysis.")

# Sidebar
st.sidebar.markdown("<h2 class='sub-header'>Settings</h2>", unsafe_allow_html=True)

# Symbol input
symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL").upper()

# Time range selection
time_range = st.sidebar.selectbox(
    "Select Time Range",
    options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=2  # Default to 6mo
)

# Time interval selection
time_interval = st.sidebar.selectbox(
    "Select Time Interval",
    options=["1d", "1wk", "1mo"],
    index=0  # Default to 1d
)

# Add yfinance installation message
st.sidebar.markdown("---")
st.sidebar.markdown("### First-time Setup")
st.sidebar.info("If you're running this app for the first time, you may need to install the yfinance package. Run this in your terminal or command prompt:\n\n```pip install yfinance```")

# Function to fetch data and calculate indicators
def fetch_and_analyze():
    with st.spinner('Fetching data and calculating indicators...'):
        # Initialize our modules
        fetcher = DataFetcher()
        ta = TechnicalAnalysis()
        sg = SignalGenerator()
        
        # Fetch data
        data = fetcher.fetch_stock_data(symbol, interval=time_interval, range=time_range)
        
        if data is None or data.empty:
            st.error(f"No data found for symbol: {symbol}")
            return
        
        # Calculate indicators
        indicators = ta.calculate_all_indicators(data)
        
        # Generate signals
        signals = sg.generate_combined_signals(indicators)
        recommendations = sg.generate_option_recommendations(indicators, signals)
        
        # Generate report
        report = sg.generate_summary_report(symbol, indicators, recommendations)
        
        # Save to session state
        st.session_state.data = data
        st.session_state.indicators = indicators
        st.session_state.signals = signals
        st.session_state.recommendations = recommendations
        st.session_state.symbol = symbol
        st.session_state.report = report

# Analyze button
if st.sidebar.button("Analyze"):
    fetch_and_analyze()

# Display tabs
if st.session_state.data is not None:
    tab1, tab2, tab3, tab4 = st.tabs(["Signal Dashboard", "Technical Indicators", "Historical Signals", "Raw Data"])
    
    with tab1:
        # Get the latest data
        latest_date = st.session_state.recommendations.index[-1]
        latest_rec = st.session_state.recommendations.iloc[-1]
        latest_price = st.session_state.data.loc[latest_date, 'close']
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Call signal box
            call_strength = latest_rec['Call_Strength']
            call_score = latest_rec['Call_Score']
            call_class = "neutral-box"
            call_text_class = "signal-none"
            
            if call_strength == "Strong":
                call_class = "call-box"
                call_text_class = "signal-strong"
            elif call_strength == "Moderate":
                call_class = "call-box"
                call_text_class = "signal-moderate"
            elif call_strength == "Weak":
                call_class = "call-box"
                call_text_class = "signal-weak"
            
            st.markdown(f"""
            <div class='recommendation-box {call_class}'>
                <h3>Call Option Signal</h3>
                <p>Score: <span class='{call_text_class}'>{call_score:.1f}/100</span></p>
                <p>Strength: <span class='{call_text_class}'>{call_strength}</span></p>
                <p>Recommendation: <span class='{call_text_class}'>{latest_rec['Call_Recommendation']}</span></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Put signal box
            put_strength = latest_rec['Put_Strength']
            put_score = latest_rec['Put_Score']
            put_class = "neutral-box"
            put_text_class = "signal-none"
            
            if put_strength == "Strong":
                put_class = "put-box"
                put_text_class = "put-signal-strong"
            elif put_strength == "Moderate":
                put_class = "put-box"
                put_text_class = "put-signal-moderate"
            elif put_strength == "Weak":
                put_class = "put-box"
                put_text_class = "put-signal-weak"
            
            st.markdown(f"""
            <div class='recommendation-box {put_class}'>
                <h3>Put Option Signal</h3>
                <p>Score: <span class='{put_text_class}'>{put_score:.1f}/100</span></p>
                <p>Strength: <span class='{put_text_class}'>{put_strength}</span></p>
                <p>Recommendation: <span class='{put_text_class}'>{latest_rec['Put_Recommendation']}</span></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Option strategy details
        if latest_rec['Option_Expiration'] != 'N/A':
            st.markdown("<h3 class='sub-header'>Option Strategy Details</h3>", unsafe_allow_html=True)
            st.markdown(f"""
            - **Current Price:** ${latest_price:.2f}
            - **Recommended Expiration:** {latest_rec['Option_Expiration']}
            - **Strike Price Selection:** {latest_rec['Strike_Selection']}
            """)
        
        # Plot signals
        sg = SignalGenerator()
        fig = sg.plot_signals(st.session_state.indicators, st.session_state.signals, symbol)
        st.pyplot(fig)
        
        # Key indicator values
        st.markdown("<h3 class='sub-header'>Key Indicator Values</h3>", unsafe_allow_html=True)
        
        # Create three columns for indicators
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # RSI
            if 'RSI' in st.session_state.indicators.columns:
                rsi_value = st.session_state.indicators.loc[latest_date, 'RSI']
                rsi_class = "signal-none"
                if rsi_value < 30:
                    rsi_class = "signal-strong"
                    rsi_status = "Oversold"
                elif rsi_value > 70:
                    rsi_class = "put-signal-strong"
                    rsi_status = "Overbought"
                else:
                    rsi_status = "Neutral"
                
                st.markdown(f"""
                <p>RSI: <span class='indicator-value {rsi_class}'>{rsi_value:.1f}</span> ({rsi_status})</p>
                """, unsafe_allow_html=True)
            
            # MACD
            if 'MACD' in st.session_state.indicators.columns:
                macd = st.session_state.indicators.loc[latest_date, 'MACD']
                macd_signal = st.session_state.indicators.loc[latest_date, 'MACD_Signal']
                macd_hist = st.session_state.indicators.loc[latest_date, 'MACD_Hist']
                
                macd_class = "signal-strong" if macd > macd_signal else "put-signal-strong"
                macd_status = "Bullish" if macd > macd_signal else "Bearish"
                
                st.markdown(f"""
                <p>MACD: <span class='indicator-value {macd_class}'>{macd:.3f}</span> ({macd_status})</p>
                <p>Signal Line: {macd_signal:.3f}</p>
                <p>Histogram: {macd_hist:.3f}</p>
                """, unsafe_allow_html=True)
        
        with col2:
            # Stochastic
            if 'Stoch_K' in st.session_state.indicators.columns:
                stoch_k = st.session_state.indicators.loc[latest_date, 'Stoch_K']
                stoch_d = st.session_state.indicators.loc[latest_date, 'Stoch_D']
                
                stoch_class = "signal-none"
                if stoch_k < 20:
                    stoch_class = "signal-strong"
                    stoch_status = "Oversold"
                elif stoch_k > 80:
                    stoch_class = "put-signal-strong"
                    stoch_status = "Overbought"
                else:
                    stoch_status = "Neutral"
                
                st.markdown(f"""
                <p>Stochastic %K: <span class='indicator-value {stoch_class}'>{stoch_k:.1f}</span> ({stoch_status})</p>
                <p>Stochastic %D: {stoch_d:.1f}</p>
                """, unsafe_allow_html=True)
            
            # Bollinger Bands
            if 'BB_Upper' in st.session_state.indicators.columns:
                bb_upper = st.session_state.indicators.loc[latest_date, 'BB_Upper']
                bb_lower = st.session_state.indicators.loc[latest_date, 'BB_Lower']
                bb_middle = st.session_state.indicators.loc[latest_date, 'BB_Middle']
                
                bb_width = (bb_upper - bb_lower) / bb_middle
                
                if latest_price > bb_upper:
                    bb_status = "Above Upper Band"
                    bb_class = "put-signal-strong"
                elif latest_price < bb_lower:
                    bb_status = "Below Lower Band"
                    bb_class = "signal-strong"
                elif latest_price > bb_middle:
                    bb_status = "Above Middle Band"
                    bb_class = "put-signal-moderate"
                else:
                    bb_status = "Below Middle Band"
                    bb_class = "signal-moderate"
                
                st.markdown(f"""
                <p>Bollinger Position: <span class='indicator-value {bb_class}'>{bb_status}</span></p>
                <p>Band Width: {bb_width:.3f}</p>
                """, unsafe_allow_html=True)
        
        with col3:
            # Moving Averages
            if 'SMA_50' in st.session_state.indicators.columns:
                sma_50 = st.session_state.indicators.loc[latest_date, 'SMA_50']
                sma_200 = st.session_state.indicators.loc[latest_date, 'SMA_200']
                
                if latest_price > sma_50 > sma_200:
                    ma_status = "Bullish (Price > 50 SMA > 200 SMA)"
                    ma_class = "signal-strong"
                elif latest_price < sma_50 < sma_200:
                    ma_status = "Bearish (Price < 50 SMA < 200 SMA)"
                    ma_class = "put-signal-strong"
                elif sma_50 > sma_200:
                    ma_status = "Bullish Trend (50 SMA > 200 SMA)"
                    ma_class = "signal-moderate"
                else:
                    ma_status = "Bearish Trend (50 SMA < 200 SMA)"
                    ma_class = "put-signal-moderate"
                
                st.markdown(f"""
                <p>Moving Averages: <span class='indicator-value {ma_class}'>{ma_status}</span></p>
                <p>50-day SMA: {sma_50:.2f}</p>
                <p>200-day SMA: {sma_200:.2f}</p>
                """, unsafe_allow_html=True)
            
            # MFI
            if 'MFI' in st.session_state.indicators.columns:
                mfi_value = st.session_state.indicators.loc[latest_date, 'MFI']
                
                mfi_class = "signal-none"
                if mfi_value < 20:
                    mfi_class = "signal-strong"
                    mfi_status = "Oversold"
                elif mfi_value > 80:
                    mfi_class = "put-signal-strong"
                    mfi_status = "Overbought"
                else:
                    mfi_status = "Neutral"
                
                st.markdown(f"""
                <p>Money Flow Index: <span class='indicator-value {mfi_class}'>{mfi_value:.1f}</span> ({mfi_status})</p>
                """, unsafe_allow_html=True)
    
    with tab2:
        # Technical indicators charts
        st.markdown("<h3 class='sub-header'>Technical Indicators</h3>", unsafe_allow_html=True)
        
        # Create plotly figure with subplots
        fig = make_subplots(rows=5, cols=1, 
                           shared_xaxes=True, 
                           vertical_spacing=0.05,
                           subplot_titles=("Price and Bollinger Bands", "RSI", "MACD", "Stochastic", "MFI"),
                           row_heights=[0.3, 0.15, 0.15, 0.15, 0.15])
        
        # Add price and Bollinger Bands
        fig.add_trace(go.Scatter(
            x=st.session_state.indicators.index, 
            y=st.session_state.indicators['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=st.session_state.indicators.index, 
            y=st.session_state.indicators['BB_Upper'],
            mode='lines',
            name='Upper BB',
            line=dict(color='gray', dash='dash')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=st.session_state.indicators.index, 
            y=st.session_state.indicators['BB_Middle'],
            mode='lines',
            name='Middle BB',
            line=dict(color='gray')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=st.session_state.indicators.index, 
            y=st.session_state.indicators['BB_Lower'],
            mode='lines',
            name='Lower BB',
            line=dict(color='gray', dash='dash'),
            fill='tonexty',
            fillcolor='rgba(200, 200, 200, 0.2)'
        ), row=1, col=1)
        
        # Add SMA lines
        fig.add_trace(go.Scatter(
            x=st.session_state.indicators.index, 
            y=st.session_state.indicators['SMA_50'],
            mode='lines',
            name='50-day SMA',
            line=dict(color='orange')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=st.session_state.indicators.index, 
            y=st.session_state.indicators['SMA_200'],
            mode='lines',
            name='200-day SMA',
            line=dict(color='red')
        ), row=1, col=1)
        
        # Add RSI
        fig.add_trace(go.Scatter(
            x=st.session_state.indicators.index, 
            y=st.session_state.indicators['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple')
        ), row=2, col=1)
        
        # Add RSI reference lines
        fig.add_shape(
            type="line", line=dict(color="red", width=1, dash="dash"),
            y0=70, y1=70, x0=st.session_state.indicators.index[0], x1=st.session_state.indicators.index[-1],
            row=2, col=1
        )
        
        fig.add_shape(
            type="line", line=dict(color="green", width=1, dash="dash"),
            y0=30, y1=30, x0=st.session_state.indicators.index[0], x1=st.session_state.indicators.index[-1],
            row=2, col=1
        )
        
        # Add MACD
        fig.add_trace(go.Scatter(
            x=st.session_state.indicators.index, 
            y=st.session_state.indicators['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='blue')
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=st.session_state.indicators.index, 
            y=st.session_state.indicators['MACD_Signal'],
            mode='lines',
            name='Signal Line',
            line=dict(color='red')
        ), row=3, col=1)
        
        # Add MACD histogram
        colors = ['green' if val >= 0 else 'red' for val in st.session_state.indicators['MACD_Hist']]
        fig.add_trace(go.Bar(
            x=st.session_state.indicators.index, 
            y=st.session_state.indicators['MACD_Hist'],
            name='Histogram',
            marker_color=colors
        ), row=3, col=1)
        
        # Add Stochastic
        fig.add_trace(go.Scatter(
            x=st.session_state.indicators.index, 
            y=st.session_state.indicators['Stoch_K'],
            mode='lines',
            name='%K',
            line=dict(color='blue')
        ), row=4, col=1)
        
        fig.add_trace(go.Scatter(
            x=st.session_state.indicators.index, 
            y=st.session_state.indicators['Stoch_D'],
            mode='lines',
            name='%D',
            line=dict(color='red')
        ), row=4, col=1)
        
        # Add Stochastic reference lines
        fig.add_shape(
            type="line", line=dict(color="red", width=1, dash="dash"),
            y0=80, y1=80, x0=st.session_state.indicators.index[0], x1=st.session_state.indicators.index[-1],
            row=4, col=1
        )
        
        fig.add_shape(
            type="line", line=dict(color="green", width=1, dash="dash"),
            y0=20, y1=20, x0=st.session_state.indicators.index[0], x1=st.session_state.indicators.index[-1],
            row=4, col=1
        )
        
        # Add MFI
        fig.add_trace(go.Scatter(
            x=st.session_state.indicators.index, 
            y=st.session_state.indicators['MFI'],
            mode='lines',
            name='MFI',
            line=dict(color='orange')
        ), row=5, col=1)
        
        # Add MFI reference lines
        fig.add_shape(
            type="line", line=dict(color="red", width=1, dash="dash"),
            y0=80, y1=80, x0=st.session_state.indicators.index[0], x1=st.session_state.indicators.index[-1],
            row=5, col=1
        )
        
        fig.add_shape(
            type="line", line=dict(color="green", width=1, dash="dash"),
            y0=20, y1=20, x0=st.session_state.indicators.index[0], x1=st.session_state.indicators.index[-1],
            row=5, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text=f"Technical Indicators for {symbol}",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Set y-axis ranges
        fig.update_yaxes(range=[0, 100], row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=4, col=1)
        fig.update_yaxes(range=[0, 100], row=5, col=1)
        
        # Display the figure
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Historical signals
        st.markdown("<h3 class='sub-header'>Historical Signals</h3>", unsafe_allow_html=True)
        
        # Create a dataframe with the signals
        historical_signals = pd.DataFrame({
            'Date': st.session_state.recommendations.index,
            'Close': st.session_state.recommendations['Close'],
            'Call Score': st.session_state.recommendations['Call_Score'].round(1),
            'Call Strength': st.session_state.recommendations['Call_Strength'],
            'Call Recommendation': st.session_state.recommendations['Call_Recommendation'],
            'Put Score': st.session_state.recommendations['Put_Score'].round(1),
            'Put Strength': st.session_state.recommendations['Put_Strength'],
            'Put Recommendation': st.session_state.recommendations['Put_Recommendation']
        })
        
        # Display the dataframe
        st.dataframe(historical_signals.sort_index(ascending=False), use_container_width=True)
        
        # Plot historical signal scores
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=st.session_state.recommendations.index,
            y=st.session_state.recommendations['Call_Score'],
            mode='lines',
            name='Call Score',
            line=dict(color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=st.session_state.recommendations.index,
            y=st.session_state.recommendations['Put_Score'],
            mode='lines',
            name='Put Score',
            line=dict(color='red')
        ))
        
        # Add threshold lines
        fig.add_shape(
            type="line", line=dict(color="green", width=1, dash="dash"),
            y0=80, y1=80, x0=st.session_state.recommendations.index[0], x1=st.session_state.recommendations.index[-1],
            annotation=dict(text="Strong Signal")
        )
        
        fig.add_shape(
            type="line", line=dict(color="orange", width=1, dash="dash"),
            y0=60, y1=60, x0=st.session_state.recommendations.index[0], x1=st.session_state.recommendations.index[-1],
            annotation=dict(text="Moderate Signal")
        )
        
        fig.add_shape(
            type="line", line=dict(color="red", width=1, dash="dash"),
            y0=40, y1=40, x0=st.session_state.recommendations.index[0], x1=st.session_state.recommendations.index[-1],
            annotation=dict(text="Weak Signal")
        )
        
        # Update layout
        fig.update_layout(
            title=f"Historical Signal Scores for {symbol}",
            xaxis_title="Date",
            yaxis_title="Signal Score",
            yaxis=dict(range=[0, 100]),
            height=500
        )
        
        # Display the figure
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Raw data
        st.markdown("<h3 class='sub-header'>Raw Data</h3>", unsafe_allow_html=True)
        
        # Create tabs for different data types
        data_tab1, data_tab2, data_tab3 = st.tabs(["Price Data", "Indicators", "Signals"])
        
        with data_tab1:
            # Display price data
            st.dataframe(st.session_state.data.sort_index(ascending=False), use_container_width=True)
            
            # Download button
            csv = st.session_state.data.to_csv().encode('utf-8')
            st.download_button(
                label="Download Price Data as CSV",
                data=csv,
                file_name=f"{symbol}_price_data.csv",
                mime="text/csv",
            )
        
        with data_tab2:
            # Display indicators data
            st.dataframe(st.session_state.indicators.sort_index(ascending=False), use_container_width=True)
            
            # Download button
            csv = st.session_state.indicators.to_csv().encode('utf-8')
            st.download_button(
                label="Download Indicators Data as CSV",
                data=csv,
                file_name=f"{symbol}_indicators.csv",
                mime="text/csv",
            )
        
        with data_tab3:
            # Display signals data
            st.dataframe(st.session_state.signals.sort_index(ascending=False), use_container_width=True)
            
            # Download button
            csv = st.session_state.signals.to_csv().encode('utf-8')
            st.download_button(
                label="Download Signals Data as CSV",
                data=csv,
                file_name=f"{symbol}_signals.csv",
                mime="text/csv",
            )
else:
    # Display instructions when no data is loaded
    st.info("Enter a stock symbol and click 'Analyze' to generate option trading signals.")
    
    # Example image
    st.markdown("### How It Works")
    st.markdown("""
    This tool helps determine optimal entry points for call or put options by analyzing technical indicators:
    
    1. **Data Collection**: Fetches historical price data for the selected stock
    2. **Technical Analysis**: Calculates key indicators like RSI, MACD, Bollinger Bands, etc.
    3. **Signal Generation**: Combines indicators to generate weighted signals for calls and puts
    4. **Recommendations**: Provides specific option strategies based on signal strength
    
    The tool uses a comprehensive algorithm that considers multiple factors to generate reliable signals.
    """)

# Footer
st.markdown("---")
st.markdown("### Disclaimer")
st.markdown("""
This tool is for informational purposes only and does not constitute financial advice. 
Always conduct your own research and consider your risk tolerance before making investment decisions.
""")
