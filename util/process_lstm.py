import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def process_lstm(df_arg: pd.DataFrame):
    df = df_arg.copy()
    df.reset_index(inplace=True)
    # close_values = df['Close'].values.reshape((1, -1))

    modelo = load_model('db/lstm')

    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler = scaler.fit(close_values)

    '''
    y = df['Close'].fillna(method='ffill')
    y = y.values.reshape(-1, 1)

    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(y)
    y = scaler.transform(y)

    # generate the input and output sequences
    n_lookback = 5  # length of input sequences (lookback period)
    n_forecast = 10  # length of output sequences (forecast period)

    X = []
    Y = []

    for i in range(n_lookback, len(y) - n_forecast + 1):
        X.append(y[i - n_lookback: i])
        Y.append(y[i: i + n_forecast])

    X = np.array(X)
    Y = np.array(Y)

    # generate the forecasts
    X_ = y[- n_lookback:]  # last available input sequence
    X_ = X_.reshape(1, n_lookback, 1)

    # st.dataframe(X_.flatten())

    Y_ = []

    for _ in range(n_forecast):
        out = modelo.predict(X_)[0][0]
        Y_ = np.append(Y_, out)

    Y_ = Y_[n_lookback - 1:]


    st.dataframe(Y_)


    # Y_ = modelo.predict(X_).reshape(-1, 1)
    Y_ = scaler.inverse_transform(Y_)

    # organize the results in a data frame
    df_past = df[['Close']].reset_index()
    df_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
    df_past['Date'] = pd.to_datetime(df_past['Date'])
    df_past['Forecast'] = np.nan
    df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

    st.dataframe(df_past.tail())
    st.dataframe(Y_.flatten())

    df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
    df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
    df_future['Forecast'] = Y_.flatten()
    df_future['Actual'] = np.nan

    results = df_past.append(df_future).set_index('Date')

    st.dataframe(results)
    '''
    
    num_prediction = 9
    forecast = lstm_predict(num_prediction, df['Close'].values.reshape((-1)), modelo)
    forecast_dates = lstm_predict_dates(df['Date'], num_prediction)

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

    results2023 =  results.loc['2023-01-01':]
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
        # x_ = scaler.inverse_transform(x)
        out = modelo.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)

    prediction_list = prediction_list[look_back - 1:]

    return prediction_list

def lstm_predict_dates(datas, num_prediction):
    last_date = datas.values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction + 1).tolist()

    return prediction_dates