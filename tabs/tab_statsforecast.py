import streamlit as st
from tabs.tab import TabInterface
from util.process_statsforecast import process_statsforecast

class TabStatsForecast(TabInterface):
    def __init__(self, tab, df):
        self.df = df

        with tab:
            self.process_tab()
    
    def process_tab(self):
        st.write('## StatsForecast')

        process_statsforecast(self.df)