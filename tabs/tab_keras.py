import streamlit as st
from tabs.tab import TabInterface

class TabKeras(TabInterface):
    def __init__(self, tab):
        with tab:
            self.process_tab()
    
    def process_tab(self):
        st.write('## Keras LSTM')

        with st.container():
            col0, col1 = st.columns([1, 1])

            with col0: 
                st.subheader('Keras LSTM')