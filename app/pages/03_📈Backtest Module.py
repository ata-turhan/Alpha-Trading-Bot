import base64

import create_backtest as cb

pass
import numpy as np
import quantstats as qs
import streamlit as st
import yfinance as yf


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

if "data" not in st.session_state:
    st.session_state["data"] = None
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
    "<h1 style='text-align: center; color: black;'> 📈 Backtest Module </h1> <br> <br>",
    unsafe_allow_html=True,
)
st.markdown("<br> <br>", unsafe_allow_html=True)
if (
    st.session_state["data"] is None
    or st.session_state["predictions"] is None
    or st.session_state["ticker"] == ""
):
    st.error("Please get the data and create the strategy first.")
else:
    ticker = st.session_state["ticker"]
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        show_initial_configuration = st.checkbox("Show initial configuration")
    with col2:
        show_tables = st.checkbox("Show tables")
    with col3:
        show_charts = st.checkbox("Show charts")
    with col4:
        show_time = st.checkbox("Show time")
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("See extra configuration for backtest"):
        col1, col2 = st.columns([1, 1])
        with col1:
            benchmark_ticker = st.text_input(
                "Enter the benchmark ticker (Default is SPY)", "SPY"
            )
            st.session_state["backtest_configuration"][
                "benchmark_ticker"
            ] = benchmark_ticker
        data = st.session_state["data"]
        predictions = np.array(st.session_state["predictions"])
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            hold_label = st.number_input(
                "Enter the hold label", value=0, step=1
            )
            st.session_state["backtest_configuration"][
                "hold_label"
            ] = hold_label
        with col2:
            buy_label = st.number_input("Enter the buy label", value=1, step=1)
            st.session_state["backtest_configuration"]["buy_label"] = buy_label
        with col3:
            sell_label = st.number_input(
                "Enter the sell label", value=2, step=1
            )
            st.session_state["backtest_configuration"][
                "sell_label"
            ] = sell_label
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            initial_capital = st.number_input(
                "Enter the initial capital", value=1000.0, step=1000.0
            )
            st.session_state["backtest_configuration"][
                "initial_capital"
            ] = initial_capital
        with col2:
            risk_free_rate = (
                st.number_input(
                    "Enter the risk free rate", value=0.01, step=0.01
                )
                / 252
            )
            st.session_state["backtest_configuration"][
                "risk_free_rate"
            ] = risk_free_rate
        with col3:
            commission = st.number_input(
                "Enter the commission rate", value=1.0, step=1.0
            )
            st.session_state["backtest_configuration"][
                "commission"
            ] = commission
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            alpha = st.number_input("Enter the alpha", value=0.05, step=0.01)
            st.session_state["backtest_configuration"]["alpha"] = alpha
        with col2:
            threshold = st.number_input(
                "Enter the threshold", value=0.0, step=1.0
            )
            st.session_state["backtest_configuration"]["threshold"] = threshold
        with col3:
            order = st.number_input("Enter the order", value=1.0, step=1.0)
            st.session_state["backtest_configuration"]["order"] = order
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            order_type = st.selectbox(
                "Select the order type", ["market", "limit"]
            )
            st.session_state["backtest_configuration"][
                "order_type"
            ] = order_type
        with col2:
            miss_rate = st.number_input(
                "Enter the trade miss rate", value=10, step=1
            )
            st.session_state["backtest_configuration"]["miss_rate"] = miss_rate
        with col3:
            precision_point = st.number_input(
                "Enter the precision point to show decimal numbers",
                value=3,
                step=1,
            )
            st.session_state["backtest_configuration"][
                "precision_point"
            ] = precision_point
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            short = st.checkbox("Use short positions")
            st.session_state["backtest_configuration"]["short"] = short
        with col2:
            trailing_take_profit = st.checkbox("Use trailing take profit")
            st.session_state["backtest_configuration"][
                "trailing_take_profit"
            ] = trailing_take_profit
        with col3:
            trailing_stop_loss = st.checkbox("Use trailing stop loss")
            st.session_state["backtest_configuration"][
                "trailing_stop_loss"
            ] = trailing_stop_loss
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            short_fee = st.number_input(
                "Enter the short fee (Default is 1)", value=1.0, step=1.0
            )
            st.session_state["backtest_configuration"]["short_fee"] = short_fee
        with col2:
            take_profit = st.number_input(
                "Enter the take profit percentage (Default is 10)",
                value=5.0,
                step=1.0,
            )
            st.session_state["backtest_configuration"][
                "take_profit"
            ] = take_profit
        with col3:
            stop_loss = st.number_input(
                "Enter the stop loss percentage (Default is 10)",
                value=5.0,
                step=1.0,
            )
            st.session_state["backtest_configuration"]["stop_loss"] = stop_loss
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            leverage = st.number_input(
                "Enter the leverage multiplier (Default is 1)",
                value=1.0,
                step=1.0,
            )
            st.session_state["backtest_configuration"]["leverage"] = leverage
        with col2:
            pass
        with col3:
            pass

        st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Run the backtest"):
        st.session_state["backtest_configuration_ready"] = True
        metrics = cb.financial_evaluation(
            hold_label,
            buy_label,
            sell_label,
            ticker,
            benchmark_ticker,
            data,
            predictions,
            risk_free_rate,
            initial_capital,
            commission,
            alpha,
            threshold,
            order,
            order_type,
            short,
            short_fee,
            trailing_take_profit,
            take_profit,
            trailing_stop_loss,
            stop_loss,
            leverage,
            miss_rate,
            show_initial_configuration,
            show_tables,
            show_charts,
            show_time,
            precision_point,
        )
        bench = yf.download(
            "SPY",
            start=data.index[0],
            end=data.index[-1],
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
        # st.dataframe(bench)
        bench["date"] = bench.index
        bench["date"] = bench["date"].dt.tz_localize(None)
        bench.index = bench["date"]
        metrics = qs.reports.metrics(
            returns=data["Adj Close"],
            benchmark=bench["Adj Close"],
            rf=0.0,
            display=False,
            mode="full",
            sep=False,
            compounded=True,
        )
        st.dataframe(metrics)
        # st.dataframe(bench)
        snapshot = qs.plots.snapshot(
            data["Adj Close"],
            title=f"{st.session_state['ticker']} Performance",
            show=False,
            mode="full",
        )
        st.write(snapshot)
        heatmap = qs.plots.monthly_heatmap(
            data["Adj Close"],
            show=False,
        )
        st.write(heatmap)
        returns = qs.plots.returns(
            data["Adj Close"],
            bench["Adj Close"],
            show=False,
        )
        st.write(returns)
        log_returns = qs.plots.log_returns(
            data["Adj Close"],
            bench["Adj Close"],
            show=False,
        )
        st.write(log_returns)
