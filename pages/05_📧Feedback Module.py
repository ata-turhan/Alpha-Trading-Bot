import streamlit as st
import datetime as dt
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from create_data import *
from backtest import *
import base64


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

st.set_page_config(page_title='Trading Bot', 
page_icon='🤖', layout="wide")

add_bg_from_local('data/background.png')

for _ in range(22):
    st.sidebar.text("\n")
st.sidebar.write('Developed by Ata Turhan')
st.sidebar.write('Contact at ataturhan21@gmail.com')

st.markdown("<h1 style='text-align: center; color: black;'> 📧 Feedback Module </h1> <br> <br>", unsafe_allow_html=True)
st.markdown("<br> <br>", unsafe_allow_html=True)
st.header("You can use the text area below to send your feedback about the app to the developer. Thanks!")

contact_form = """
<form action="https://formsubmit.co/kuantum21fizik@gmail.com" method="POST">
     <input type="hidden" name="_captcha" value="false">
     <input type="hidden" name="_subject" value="Trading Bot Feedback!">
     <input type="text" name="name" placeholder="Your name" required>
     <input type="email" name="email" placeholder="Your email" required>
     <textarea name="message" placeholder="Your message here"></textarea>
     <button type="submit">Send</button>
     <input type="hidden" name="_next" value="http://localhost:8501">
</form>
"""

st.markdown(contact_form, unsafe_allow_html=True)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")


        