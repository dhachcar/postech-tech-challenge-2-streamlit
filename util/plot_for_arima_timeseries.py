import pandas as pd
import plotly.graph_objs as go
import streamlit as st

def plot(df: pd.DataFrame, y, title: str):
  media = df.rolling(7).mean()
  std = df.rolling(7).std()

  fig = go.Figure()
  fig.add_trace(go.Scatter(x=df.index, y=df[y], name='Preço de fechamento'))
  fig.add_trace(go.Scatter(x=media.index, y=media[y], name='Média móvel'))
  fig.add_trace(go.Scatter(x=std.index, y=std[y], name='Desvio padrão'))
  fig.update_layout(title=title)
  st.plotly_chart(fig)