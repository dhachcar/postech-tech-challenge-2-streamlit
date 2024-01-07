
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, Naive, SeasonalNaive, WindowAverage, SeasonalWindowAverage
from statsforecast.utils import ConformalIntervals
import streamlit as st
import global_config

from util.plot_for_arima_timeseries import plot

def process_statsforecast(df: pd.DataFrame):
    # chaveador para processar no app ou carregar as previsões do colabs
    if not global_config.use_colabs_statsforecast_export:
        df.reset_index(inplace=True)
        df.rename(columns={'Date':'ds', 'Close':'y'}, inplace=True)
        df['unique_id'] = df.index + 1

        df = df[['unique_id', 'ds', 'y']]
        df.unique_id = df.unique_id.astype(str)
        df.unique_id = 'IBOVESPA'

        df_treino = df.query('"2020-01-01" <= ds <= "2023-10-01"')
        df_validacao = df.query('"2023-10-02" <= ds <= "2023-12-15"')

        df_treino = preenche_dias_faltantes(df_treino)
        df_validacao = preenche_dias_faltantes(df_validacao)
        
        intervals = ConformalIntervals(h=60, n_windows=4)
        
        sf = StatsForecast(
            models = [
            Naive(),
            SeasonalNaive(season_length=60),
            WindowAverage(window_size=60, prediction_intervals=intervals),
            SeasonalWindowAverage(window_size=7, season_length=60, prediction_intervals=intervals),
            AutoARIMA(season_length = 60)
            ],
            freq = 'D',
            n_jobs=-1
        )

        sf.fit(df_treino)

        df_forecast = sf.predict(h=60, level=[70, 80, 90])
        df_forecast.reset_index(inplace=True)
        df_forecast_merged = df_forecast.merge(df_validacao, how='left', on=['unique_id', 'ds'])
        df_forecast_merged.dropna(inplace=True)
        df_forecast = df_forecast_merged
    else:
        # carrega os dataframes processados no colabs
        df_treino = pd.read_csv('db/statsforecast/statsforecast-treino.csv')
        df_forecast = pd.read_csv('db/statsforecast/statsforecast-forecast.csv')

    plot_last_days = 180
    niveis_de_confiaca = [70, 80, 90]

    plot_naive = StatsForecast.plot(df_treino.tail(plot_last_days), df_forecast, models=['Naive'], level=niveis_de_confiaca, engine='plotly')
    plot_seasonal_naive = StatsForecast.plot(df_treino.tail(plot_last_days), df_forecast, models=['SeasonalNaive'], level=niveis_de_confiaca, engine='plotly')
    plot_auto_arima = StatsForecast.plot(df_treino.tail(plot_last_days), df_forecast, models=['AutoARIMA'], level=niveis_de_confiaca, engine='plotly')
    plot_window_average = StatsForecast.plot(df_treino.tail(plot_last_days), df_forecast, models=['WindowAverage'], level=niveis_de_confiaca, engine='plotly')
    plot_seas_wa = StatsForecast.plot(df_treino.tail(plot_last_days), df_forecast, models=['SeasWA'], level=niveis_de_confiaca, engine='plotly')

    st.plotly_chart(plot_naive)
    st.plotly_chart(plot_seasonal_naive)
    st.plotly_chart(plot_auto_arima)
    st.plotly_chart(plot_window_average)
    st.plotly_chart(plot_seas_wa)

def preenche_dias_faltantes(df: pd.DataFrame) -> pd.DataFrame :
    clone = df.copy()
    clone['ds'] = pd.to_datetime(clone['ds'])
    clone = clone.set_index('ds')

    clone['y'] = clone['y'].resample('D').mean()
    clone.reset_index(inplace=True)
    clone.fillna(method='ffill', inplace=True)
    clone['unique_id'] = 'IBOVESPA'

    return clone