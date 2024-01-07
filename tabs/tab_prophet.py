import streamlit as st
from tabs.tab import TabInterface

class TabProphet(TabInterface):
    def __init__(self, tab):
        with tab:
            self.process_tab()
    
    def process_tab(self):
        st.write('## Meta Prophet')

        with st.container():
            st.image('assets/img/tab-prophet/1.png')
                