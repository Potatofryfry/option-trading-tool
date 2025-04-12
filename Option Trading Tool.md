# Option Trading Tool

A web-based tool to help determine optimal entry points for call or put options based on technical analysis.

## Features

- **Signal Dashboard**: Get clear recommendations for call and put options with confidence scores
- **Technical Indicators**: View interactive charts of RSI, MACD, Bollinger Bands, and more
- **Historical Signals**: Track how signals have evolved over time
- **Raw Data Access**: Download price data, indicators, and signals for further analysis

## Quick Start

This repository is ready to deploy directly to Streamlit Cloud with no coding required!

### One-Click Deployment

1. **Fork this repository**:
   - Click the "Fork" button at the top right of this page
   - This creates a copy of the repository in your GitHub account

2. **Deploy to Streamlit Cloud**:
   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Sign in with your GitHub account
   - Click "New app"
   - Select this repository from the dropdown
   - Set the main file path to "app.py"
   - Click "Deploy"

3. **Access your app**:
   - After deployment (takes about 1-2 minutes), you'll get a URL to access your app
   - The app will be available at: https://yourusername-option-trading-tool.streamlit.app

## Using the Tool

1. Enter a stock symbol in the sidebar (e.g., AAPL, MSFT, GOOGL)
2. Select the time range and interval for analysis
3. Click "Analyze" to generate option trading signals
4. View the results in the different tabs

## Understanding the Signals

### Call Option Signals:
- **Strong Signal (Score 80-100)**: Consider buying near-term call options (2-4 weeks expiration)
- **Moderate Signal (Score 60-79)**: Consider mid-term call options (1-2 months expiration)
- **Weak Signal (Score 40-59)**: Monitor for call opportunities or consider longer-term options

### Put Option Signals:
- **Strong Signal (Score 80-100)**: Consider buying near-term put options (2-4 weeks expiration)
- **Moderate Signal (Score 60-79)**: Consider mid-term put options (1-2 months expiration)
- **Weak Signal (Score 40-59)**: Monitor for put opportunities or consider longer-term options

## Disclaimer

This tool is for informational purposes only and does not constitute financial advice. Always conduct your own research and consider your risk tolerance before making investment decisions.
