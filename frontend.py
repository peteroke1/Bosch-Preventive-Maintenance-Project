import streamlit as st
import requests
import pandas as pd
import base64
from PIL import Image
import io
import re

st.set_page_config(
    page_title="Bosch Rexroth - Predictive Maintenance"
    page_icon=None
    layout="wide"
    initial_sidebar_state="collapsed"
)

#CONFIG
API_BASE = "http://localhost:8000"

# STYLING
st.markdown("""
<style>
    
/*Main background*/
.main {
    background-color: #0e1117;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
    padding: 20px;
}
            
/* Sidebar title */
.sidebar-title {
    font-size: 20px;
    font-weight: 700;
    color: white;
    margin-bottom: 20px;
}
            
/* Navigation */
            
            """)