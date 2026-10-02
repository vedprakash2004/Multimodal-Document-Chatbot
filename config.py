import os

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI


load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def configure_page():
    st.set_page_config(
        page_title="Multimodal RAG Q&A",
        page_icon="🤖",
        layout="wide"
    )


def check_api_keys():

    if not GROQ_API_KEY:
        st.error("❌ GROQ_API_KEY not found in .env file")
        st.stop()

    if not OPENAI_API_KEY:
        st.error("❌ OPENAI_API_KEY not found in .env file")
        st.stop()


check_api_keys()


groq = Groq(
    api_key=GROQ_API_KEY
)


openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)