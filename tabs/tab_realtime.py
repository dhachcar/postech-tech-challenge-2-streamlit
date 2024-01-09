import streamlit as st
from tabs.tab import TabInterface
from util.process_lstm import process_lstm

class TabRealtime(TabInterface):
    def __init__(self, tab, df):
        self.df = df

        with tab:
            self.process_tab()
    
    def process_tab(self):
        st.write('## Realtime')

        with st.spinner('Processando...'):
            x = 1
            # process_lstm(self.df)

        st.success('Processamento concluído', icon='✅')