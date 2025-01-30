from dotenv import load_dotenv
import os
import logging
import streamlit as st
load_dotenv()





GPT_CONFIG = {
    'api_key': st.secrets['OPENAI_API_KEY'],
    'api_base': st.secrets['OPENAI_API_BASE'],
    'api_version': st.secrets['GPT_API_VERSION'],
    'model': st.secrets['GPT_MODEL'],
    'deployment_name': st.secrets['GPT_DEPLOYMENT_NAME']
}



