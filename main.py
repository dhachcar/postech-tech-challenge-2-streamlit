import pandas as pd
import streamlit as st
import yfinance as yf
import datetime
import plotly.graph_objs as go
from tabs.tab_keras import TabKeras
from tabs.tab_prophet import TabProphet
from tabs.tab_realtime import TabRealtime
from tabs.tab_statsforecast import TabStatsForecast
from tabs.tab_statsmodels import TabStatsModels
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(layout="wide")

min_date = datetime.date(2020, 1, 1)
max_date = datetime.date(2023, 12, 31)
df_bvsp = yf.download('^BVSP', start=min_date, end=max_date)
df_bvsp = pd.DataFrame(df_bvsp, columns=['Close'])

with st.container():
    _, col1, col2, _ = st.columns([2, 2, 4, 2])

    with col1:
        st.subheader('Últimos 10 fechamentos')
        st.dataframe(df_bvsp.tail(10))

    with col2:
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
TabRealtime(tab4, df_bvsp)