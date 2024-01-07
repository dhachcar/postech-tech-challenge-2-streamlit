import streamlit as st
from tabs.tab import TabInterface
from util.process_prophet import process_prophet

class TabProphet(TabInterface):
    def __init__(self, tab, df):
        self.df = df

        with tab:
            self.process_tab()
    
    def process_tab(self):
        st.write('## Meta Prophet')
        
        with st.spinner('Processando...'):
            process_prophet(self.df)

        st.success('Processamento concluído', icon='✅')
        