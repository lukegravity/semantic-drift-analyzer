import streamlit as st

def sidebar():
    st.sidebar.header("⚙️ Configuration")
    alpha = st.sidebar.slider("Traffic Weight (α)", 0.0, 1.0, 0.6, 0.05)
    beta = st.sidebar.slider("Inlinks Weight (β)", 0.0, 1.0, 0.3, 0.05)
    gamma = st.sidebar.slider("Depth Weight (γ)", 0.0, 1.0, 0.1, 0.05)
    st.sidebar.markdown("---")
    highlight_navboost = st.sidebar.checkbox("Highlight NavBoost Drift", True)
    return alpha, beta, gamma, highlight_navboost
