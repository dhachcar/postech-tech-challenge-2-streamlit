import streamlit as st
from tabs.tab import TabInterface
from util.process_lstm import process_lstm
from util.process_realtime import process_realtime

class TabRealtime(TabInterface):
    def __init__(self, tab, df):
        self.df = df

        with tab:
            self.process_tab()
    
    def process_tab(self):
        st.write('## Realtime')

        with st.spinner('Processando...'):
            process_realtime(self.df)

        st.success('Processamento concluído', icon='✅')