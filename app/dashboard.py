import streamlit as st
import yfinance as yf
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

st.title('MarketPulse AI: Predictive Analytics for Market Insights')
ticker = st.text_input('Enter Stock Symbol','AAPL')
if st.button('Predict'):
    df = yf.download(ticker, period='1y')
    data = df[['Close']]
    st.line_chart(data)
    st.success('Prediction demo complete - integrate model for real use!')
