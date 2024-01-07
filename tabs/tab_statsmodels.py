import streamlit as st
from tabs.tab import TabInterface
from util.process_statsmodels import process_statsmodels

class TabStatsModels(TabInterface):
    def __init__(self, tab, df):
        self.df = df

        with tab:
            self.process_tab()
    
    def process_tab(self):
        st.write('## statsmodels')
        process_statsmodels(self.df)  