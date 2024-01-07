import pandas as pd
import streamlit as st
import yfinance as yf
import datetime
import plotly.graph_objs as go
import requests
import json
from tabs.tab_keras import TabKeras
from tabs.tab_prophet import TabProphet
from tabs.tab_statsforecast import TabStatsForecast
from tabs.tab_statsmodels import TabStatsModels

# https://medium.com/codex/streamlit-fastapi-%EF%B8%8F-the-ingredients-you-need-for-your-next-data-science-recipe-ffbeb5f76a92
# https://medium.com/@borandabak/predicting-stock-prices-with-lstm-a-fastapi-and-streamlit-web-application-1ad0559639b7

# st.set_page_config(layout="wide")

min_date = datetime.date(2020, 1, 1)
max_date = datetime.date(2023, 12, 31)
df_bvsp = yf.download('^BVSP', start=min_date, end=max_date)
df_bvsp = pd.DataFrame(df_bvsp, columns=['Close'])

with st.container():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_bvsp.index, y=df_bvsp['Close'], name='Close'))
    fig.update_layout(title='IBOVESPA')
    st.plotly_chart(fig)

tab0, tab1, tab2, tab3, tab4 = st.tabs(
    tabs=[':one: statsmodels', ':two: StatsForecast', ':three: Meta Prophet', ':four: Keras LSTM', ':five: Realtime'])

TabStatsModels(tab0, df_bvsp)
TabStatsForecast(tab1, df_bvsp)
TabProphet(tab2, df_bvsp)
TabKeras(tab3, df_bvsp)

















API_URL = "http://127.0.0.1:8000/LSTM_Predict"



# stock_name = st.selectbox('Please choose stock name', ('^BVSP', 'PETR4.SA', 'ITUB4.SA'))
# start_date = st.date_input("Start date", min_value=min_date, max_value=max_date, value=min_date)
# end_date = st.date_input("End date", min_value=min_date, max_value=max_date, value=max_date)

# if start_date <= end_date:
#     st.success("Start date: `{}`\n\nEnd date:`{}`".format(start_date, end_date))
# else:
#     st.error("Error: End date must be after start date.")



# stock_data.to_csv(f'{stock_name}_data.csv',index=False)


# if st.button("Predict"):
#     payload = {"stock_name": stock_name}

#     try:
#         response = requests.post(API_URL, json=payload)
#         response.raise_for_status()

#         predictions = response.json()
#         predicted_prices = predictions["prediction"]

#         actual_prices = stock_data['Close'].tolist()
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=stock_data.index, y=actual_prices, name='Actual'))
#         fig.add_trace(go.Scatter(x=stock_data.index[-len(predicted_prices):], y=predicted_prices, name='Predicted'))
#         fig.update_layout(title=f"{stock_name} Stock Price")
#         st.plotly_chart(fig)

#     except requests.exceptions.RequestException as e:
#         st.error(f"Error occurred while making the request: {e}")