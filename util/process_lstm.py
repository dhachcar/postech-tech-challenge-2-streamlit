import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
import joblib
from keras.models import load_model

def process_lstm(df_arg: pd.DataFrame):
    df = df_arg.copy()
    df.reset_index(inplace=True)

    # recarrega o lstm e o scaler
    modelo = load_model('db/lstm')
    scaler = joblib.load('db/lstm_scaler/lstm_smooth_scaler.save') 

    # "suaviza" os valores
    alpha = 0.1
    df['Close'] = df['Close'].ewm(alpha = alpha, adjust = False).mean()
    close_values_transformed = scaler.transform(df['Close'].values.reshape((-1, 1)))

    # faz previsões
    num_prediction = 7
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

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Fechamento (real)'))
    fig.add_trace(go.Scatter(x=results2023.index, y=results2023['Forecast'], name='Fechamento (previsto)'))
    fig.update_layout(title='Previsões para o IBOVESPA')
    st.plotly_chart(fig)

def lstm_predict(num_prediction, series, modelo):
    look_back = 5
    prediction_list = series[-look_back:]

    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = modelo.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)

    prediction_list = prediction_list[look_back - 1:]

    return prediction_list

def lstm_predict_dates(datas, num_prediction):
    last_date = datas.values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction + 1).tolist()

    return prediction_dates