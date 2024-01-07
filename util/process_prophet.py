import pandas as pd
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.serialize import model_from_json
import global_config

def process_prophet(df: pd.DataFrame):
    # chaveador para processar no app ou carregar as previsões do colabs
    if not global_config.use_colabs_prophet_export:
        df.reset_index(inplace=True)
        df.rename(columns={'Date':'ds', 'Close':'y'}, inplace=True)
        df['unique_id'] = df.index + 1

        train_data = df.sample(frac=0.8, random_state=0)

        modelo = Prophet(seasonality_mode='additive', daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        modelo.add_country_holidays(country_name='BR')
        modelo.fit(train_data)

        df_futuro = modelo.make_future_dataframe(periods=90, freq='D')
        previsao = modelo.predict(df_futuro)

        plot_prophet = plot_plotly(modelo, previsao)

        st.plotly_chart(plot_prophet)
    else:
        with open('db/prophet/prophet-model.json', 'r') as fin:
            modelo = model_from_json(fin.read()) 
            df_futuro = modelo.make_future_dataframe(periods=90, freq='D')
            previsao = modelo.predict(df_futuro)

            plot_prophet = plot_plotly(modelo, previsao)
            
            st.plotly_chart(plot_prophet)