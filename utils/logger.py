import streamlit as st
import time

def log(msg):
    st.text(f"[{time.strftime('%H:%M:%S')}] {msg}")