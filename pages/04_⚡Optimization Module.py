import base64

import numpy as np
import streamlit as st
import create_backtest as cb
import create_data as cd


def add_bg_from_local(background_file, sidebar_background_file):
    with open(background_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    with open(sidebar_background_file, "rb") as image_file:
        sidebar_encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/png;base64,{encoded_string.decode()});
        background-size: cover
    }}
    section[data-testid="stSidebar"] div[class="css-6qob1r e1fqkh3o3"]
    {{
        background-image: url(data:image/png;base64,{sidebar_encoded_string.decode()});
        background-size: 400px 800px
    }}
    """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Trading Bot", page_icon="🤖", layout="wide")

add_bg_from_local("data/background.png", "data/bot.png")

if "ohlcv" not in st.session_state:
    st.session_state["ohlcv"] = None
if "predictions" not in st.session_state:
    st.session_state["predictions"] = None
if "ticker" not in st.session_state:
    st.session_state["ticker"] = ""
if "backtest_configuration_ready" not in st.session_state:
    st.session_state["backtest_configuration_ready"] = False
if "backtest_configuration" not in st.session_state:
    st.session_state["backtest_configuration"] = {}


for _ in range(22):
    st.sidebar.text("\n")
st.sidebar.write("Developed by Ata Turhan")
st.sidebar.write("Contact at ataturhan21@gmail.com")

st.markdown(
    "<h1 style='text-align: center; color: black;'> ⚡Optimization Module </h1> <br> <br>",
    unsafe_allow_html=True,
)
st.markdown("<br> <br>", unsafe_allow_html=True)
if (
    st.session_state["ohlcv"] is None
    or st.session_state["predictions"] is None
    or st.session_state["ticker"] == ""
    or st.session_state["backtest_configuration_ready"] == False
):
    st.error(
        "Please get the data, create the strategy and backtest the strategy first."
    )
else:
    ticker = st.session_state["ticker"]
    ohlcv = st.session_state["ohlcv"]
    predictions = np.array(st.session_state["predictions"])
    st.markdown("<br>", unsafe_allow_html=True)
    metric_optimized = st.selectbox(
        "Please select the backtest metrics to optimize:",
        ["<Select>", "Total Return", "Sharpe Ratio", "Sortino Ratio", "Omega Ratio"],
    )
    st.markdown("<br>", unsafe_allow_html=True)
    take_profit_ranges = st.slider(
        "Select a range for take profit values", 0, 50, (5, 25)
    )
    take_profit_values = list(range(take_profit_ranges[0], take_profit_ranges[1]))
    st.markdown("<br>", unsafe_allow_html=True)
    stop_loss_ranges = st.slider("Select a range for stop loss values", 0, 50, (5, 25))
    stop_loss_values = list(range(stop_loss_ranges[0], stop_loss_ranges[1]))
    st.markdown("<br>", unsafe_allow_html=True)
    leverage_ranges = st.slider("Select a range for leverage values", 1, 20, (1, 5))
    leverage_values = list(range(leverage_ranges[0], leverage_ranges[1]))
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        iteration = int(
            st.number_input(
                "Choose the iteration number to try optimized values.", value=0
            )
        )
    with col2:
        best_n = int(
            st.number_input("Choose the number of best combinations to see.", value=0)
        )
    col1, col2 = st.columns([1, 1])
    with col1:
        verbose = st.checkbox("Optimize verbosely")
    with col2:
        show_results = st.checkbox("Show results")

    if metric_optimized == "<Select>" or iteration == 0 or best_n == 0:
        st.error("Please fill all the required fields.")

    elif st.button("Run the optimization"):
        cb.optimize_backtest(
            metric_optimized=metric_optimized,
            take_profit_values=take_profit_values,
            stop_loss_values=stop_loss_values,
            leverage_values=leverage_values,
            iteration=iteration,
            best_n=best_n,
            verbose=verbose,
            show_results=show_results,
        )
        st.success("Optimization is successful!")
        st.balloons()
