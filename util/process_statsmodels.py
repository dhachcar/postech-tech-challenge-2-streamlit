import numpy as np
import pandas as pd
import plotly.graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st

from util.plot_for_arima_timeseries import plot

def process_statsmodels(df_arg: pd.DataFrame):
    df = df_arg.copy()

    col0, col1 = st.columns([1, 1])

    with col0:
        # step 1
        df_log = np.log(df)
        plot(df_log, 'Close', '1) IBOVESPA (log)')

        # step 2
        janela_movel = 5
        df_mean = df_log.rolling(janela_movel).mean()
        df_sem_media = (df_log - df_mean).dropna()
        plot(df_sem_media, 'Close', '2) IBOVESPA (sem média)')

    with col1:
        # step 3
        df_diff = df_sem_media.diff(1).dropna()
        plot(df_diff, 'Close', '3) IBOVESPA (diff)')

        # step 4
        arima_model = ARIMA(df_diff['Close'], order=(2, 2, 2))
        arima_results = arima_model.fit()

        # step 5
        predictions = arima_results.fittedvalues
        predictions.index = df_diff.index
        predicted_values = df_log['Close'].iloc[0] + np.cumsum(predictions)

        # st.write('Previsões')
        # st.dataframe(predicted_values)

        # output final
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_log.index, y=df_log['Close'], name='Fechamento (real)'))
        fig.add_trace(go.Scatter(x=predicted_values.index, y=predicted_values, name='Fechamento (previsto)'))
        fig.update_layout(title='Previsões para o IBOVESPA')
        st.plotly_chart(fig)