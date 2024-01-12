import numpy as np
import pandas as pd
import streamlit as st
import datetime
import joblib
from keras.models import load_model
import plotly.graph_objs as go
from util.process_lstm import lstm_predict, lstm_predict_dates

def process_realtime(df_arg: pd.DataFrame):
    min = datetime.date(2023, 12, 25)
    max_date = datetime.date(2024, 1, 20)

    col0, col1, _ = st.columns([2, 4, 6])

    with col0:
        end_date = st.date_input("Data final da previsão", min_value=min, max_value=max_date, value=max_date)
        date_difference = min - end_date
        days_between = np.abs(date_difference.days)

        if st.button(":crystal_ball: Prever"):
            df = df_arg.copy()
            df.reset_index(inplace=True)

            # recarrega o lstm e o scaler
            modelo = load_model('db/lstm')
            scaler = joblib.load('db/lstm_scaler/lstm_smooth_scaler.save') 

            # "suaviza" os valores
            alpha = 0.1
            df['Close'] = df['Close'].ewm(alpha = alpha, adjust = False).mean()
            close_values_transformed = scaler.transform(df['Close'].values.reshape((-1, 1)))

            close_values_transformed = scaler.transform(df['Close'].values.reshape((-1, 1)))
        
            st.info(f'Prevendo os próximos {days_between} dias (à partir de  {min.strftime("%d/%m/%Y")})', icon='📣')
            st.warning('ATENÇÃO! Quanto maior o intervalo de tempo selecionado, maior a taxa de erro!', icon='⚠️')

            # faz previsões
            num_prediction = days_between
            forecast = lstm_predict(num_prediction, close_values_transformed, modelo)
            forecast_dates = lstm_predict_dates(df['Date'], num_prediction)

            forecast = forecast.reshape(-1, 1)
            forecast = scaler.inverse_transform(forecast)

            df_past = df[['Date', 'Close']]
            df_past.rename(columns={'Close': 'Actual'}, inplace=True)
            df_past['Date'] = pd.to_datetime(df_past['Date'])
            df_past['Forecast'] = np.nan
            df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

            df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
            df_future['Date'] = forecast_dates
            df_future['Forecast'] = forecast.flatten()
            df_future['Actual'] = np.nan

            frames = [df_past, df_future]
            results = pd.concat(frames, ignore_index=True)

            results2023 = results.loc['2023-01-01':]
            results2023 = results2023.set_index('Date')

            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Fechamento (real)'))
                fig.add_trace(go.Scatter(x=results2023.index, y=results2023['Forecast'], name='Fechamento (previsto)'))
                fig.update_layout(title='Previsões para o IBOVESPA')
                st.plotly_chart(fig)