import streamlit as st
from tabs.tab import TabInterface

class TabKeras(TabInterface):
    def __init__(self, tab, df):
        self.df = df

        with tab:
            self.process_tab()
    
    def process_tab(self):
        st.write('## Keras LSTM')

        with st.spinner('Processando...'):
            x = 1 # process_prophet(self.df)

        st.success('Processamento concluído', icon='✅')