import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from numpy import log, sqrt, exp
import requests

# Page Config
st.set_page_config(page_title="Black-Scholes Dashboard", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        .main { background-color: #f5f7fa; font-family: 'Inter', sans-serif; }
        .sidebar .sidebar-content { background-color: #ffffff; }
        
        .stButton>button {
            background-color: #4b5efa;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 8px 16px;
        }
        
        .card {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        
        .card:hover { transform: translateY(-2px); }
        
        .metric-title {
            font-size: 14px;
            color: black;
            font-weight: 500;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: 700;
            margin-top: 8px;
            color: #1e293b;
        }
        
        .call { color: #22c55e; }
        .put { color: #ef4444; }
        
        h1 {
            color: #1e293b;
            font-weight: 700;
            font-size: 32px;
            margin-bottom: 24px;
        }
        
        h2 {
            color: #334155;
            font-weight: 600;
            font-size: 20px;
            margin: 24px 0 16px;
        }
        
        .company-header {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 24px;
        }
        
        .company-logo {
            width: 48px;
            height: 48px;
            border-radius: 8px;
            object-fit: contain;
        }
    </style>
""", unsafe_allow_html=True)

# Title with company info placeholder
st.markdown("""
    <h1>Black-Scholes Option Pricing</h1>
    <div id='company-header-placeholder'></div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h2>⚙️ Parameters</h2>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style='margin: 16px 0;'>
            <a href="https://www.linkedin.com/in/-nihal-thomas/" target="_blank" style="text-decoration: none;">
                <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" alt="LinkedIn" style="width:20px; vertical-align: middle;"/>
                <span style="margin-left: 8px; color: white;">Nihal Thomas</span>
            </a>
        </div>
    """, unsafe_allow_html=True)
    
    ticker = st.text_input("Stock Ticker", value="AAPL").upper()
    strike_price = st.number_input("Strike Price (K)", value=150.0, step=1.0)
    time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.0, step=0.1)
    
    st.markdown("### Auto Calculators", unsafe_allow_html=True)
    
    if st.checkbox("Estimate Risk-Free Rate (10Y Treasury)"):
        try:
            treasury_data = yf.download("^TNX", period="5d", interval="1d")
            risk_free_rate = treasury_data['Close'].iloc[-1] / 100
            st.success(f"Risk-Free Rate: {risk_free_rate:.4f}")
        except:
            risk_free_rate = st.number_input("Risk-Free Rate (r)", value=0.05, step=0.01)
    else:
        risk_free_rate = st.number_input("Risk-Free Rate (r)", value=0.05, step=0.01)

    if st.checkbox("Estimate Volatility (Past N Days)"):
        hist_days = st.slider("Lookback Days", 30, 365, 90)
        try:
            hist_data = yf.Ticker(ticker).history(period=f"{hist_days}d")
            log_returns = np.log(hist_data['Close'] / hist_data['Close'].shift(1)).dropna()
            volatility = log_returns.std() * np.sqrt(252)
            st.success(f"Volatility: {volatility:.4f}")
        except:
            volatility = st.number_input("Volatility (σ)", value=0.2, step=0.01)
    else:
        volatility = st.number_input("Volatility (σ)", value=0.2, step=0.01)

# Fetch company data
try:
    stock = yf.Ticker(ticker)
    current_price = stock.history(period="1d")['Close'].iloc[-1]
    company_name = stock.info.get('longName', ticker)
    
    # Fetch company logo using Clearbit API (or similar service)
    domain = stock.info.get('website', '').replace('http://', '').replace('https://', '').split('/')[0]
    logo_url = f"https://logo.clearbit.com/{domain}" if domain else "https://via.placeholder.com/48"
    
    # Verify logo exists, fallback to placeholder if not
    response = requests.head(logo_url)
    if response.status_code != 200:
        logo_url = "https://via.placeholder.com/48"
        
    st.markdown(f"""
        <div class='company-header'>
            <img src='{logo_url}' class='company-logo' alt='{company_name} logo'>
            <div>
                <h2 style='margin: 0;'>{company_name}</h2>
                <span style='color: white;'>{ticker}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
except:
    st.error("Invalid Ticker or Network Issue")
    st.stop()

# Black-Scholes calculation
def black_scholes(S, K, T, sigma, r):
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    call = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    put = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call, put

call_price, put_price = black_scholes(current_price, strike_price, time_to_maturity, volatility, risk_free_rate)

# Dashboard Layout
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
        <div class='card'>
            <div class='metric-title'>Current Price</div>
            <div class='metric-value'>${current_price:.2f}</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class='card'>
            <div class='metric-title'>Strike Price</div>
            <div class='metric-value'>${strike_price:.2f}</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class='card'>
            <div class='metric-title'>Call Price</div>
            <div class='metric-value call'>${call_price:.2f}</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class='card'>
            <div class='metric-title'>Put Price</div>
            <div class='metric-value put'>${put_price:.2f}</div>
        </div>
    """, unsafe_allow_html=True)

# Heatmaps
st.markdown("<h2>📊 Price Sensitivity Analysis</h2>", unsafe_allow_html=True)
spot_prices = np.linspace(current_price * 0.8, current_price * 1.2, 10)
vols = np.linspace(volatility * 0.5, volatility * 1.5, 10)

call_matrix = np.zeros((len(vols), len(spot_prices)))
put_matrix = np.zeros((len(vols), len(spot_prices)))

for i, vol in enumerate(vols):
    for j, spot in enumerate(spot_prices):
        call, put = black_scholes(spot, strike_price, time_to_maturity, vol, risk_free_rate)
        call_matrix[i, j] = call
        put_matrix[i, j] = put

col5, col6 = st.columns(2)

with col5:
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.heatmap(call_matrix,
                xticklabels=np.round(spot_prices, 2),
                yticklabels=np.round(vols, 2),
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",
                cbar_kws={'label': 'Price ($)'})
    ax1.set_xlabel("Spot Price")
    ax1.set_ylabel("Volatility")
    st.pyplot(fig1)

with col6:
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(put_matrix,
                xticklabels=np.round(spot_prices, 2),
                yticklabels=np.round(vols, 2),
                annot=True,
                fmt=".2f",
                cmap="YlOrRd",
                cbar_kws={'label': 'Price ($)'})
    ax2.set_xlabel("Spot Price")
    ax2.set_ylabel("Volatility")
    st.pyplot(fig2)

# Footer
st.markdown("""
    <div style='text-align: center; color: #64748b; margin-top: 32px; padding: 16px;'>
        Developed by Nihal Thomas • 2025
    </div>
""", unsafe_allow_html=True)
