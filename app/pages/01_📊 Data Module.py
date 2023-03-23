import base64
import re

import create_data as cd
import pandas as pd
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

for _ in range(18):
    st.sidebar.text("\n")
st.sidebar.write("Developed by Ata Turhan")
st.sidebar.write("Contact at ataturhan21@gmail.com")


def fetch_data_button_click():
    if st.session_state["all_areas_filled"]:
        if st.session_state["fetch_data_button_clicked"] == True:
            st.session_state["show_data_button_clicked"] = False
            st.session_state["show_chart_button_clicked"] = False
            st.session_state["chart_data_selectbox_clicked"] = False
        st.session_state["fetch_data_button_clicked"] = True
        st.session_state["ohlcv"] = cd.get_financial_data(
            tickers=tickers,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
        )


def show_data_button_click():
    st.session_state["show_data_button_clicked"] = True


def show_chart_button_click():
    st.session_state["show_chart_button_clicked"] = True


def chart_data_selectbox_click():
    st.session_state["chart_data_selectbox_clicked"] = True


def clear_data():
    st.session_state["ohlcv"] = None
    st.session_state.conf_change = True
    st.session_state["show_data_button_clicked"] = False
    st.session_state["show_chart_button_clicked"] = False


st.markdown(
    "<h1 style='text-align: center; color: black;'> 📊 Data Module </h1> <br> <br>",
    unsafe_allow_html=True,
)

style = "<style>.row-widget.stButton {text-align: center;}</style>"
st.markdown(style, unsafe_allow_html=True)


col1, col2, col3 = st.columns([1, 2, 1])
data_fetch_way = col2.selectbox(
    "Which way do you want to get the prices: ",
    ["<Select>", "Fetch over the internet", "Read from a file"],
    on_change=clear_data,
)
st.markdown("<br>", unsafe_allow_html=True)


if "conf_change" not in st.session_state:
    st.session_state.conf_change = False
if "ohlcv" not in st.session_state:
    st.session_state["ohlcv"] = None
if "ticker" not in st.session_state:
    st.session_state["ticker"] = ""
if "assets" not in st.session_state:
    st.session_state["assets"] = {}
if "show_data_button_clicked" not in st.session_state:
    st.session_state["show_data_button_clicked"] = False
if "show_chart_button_clicked" not in st.session_state:
    st.session_state["show_chart_button_clicked"] = False


smooth_method = "<Select>"

stocks_and_etfs = {
    "Microsoft": "MSFT",
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Meta": "META",
    "Amazon": "AMZN",
    "S&P500": "^SPX",
}
forex = {
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "JPY=X",
    "GBP/USD": "GBPUSD=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X",
    "USD/CNY": "CNY=X",
}
crypto = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "BNB": "BNB-USD",
    "XRP": "XRP-USD",
    "Cardano": "ADA-USD",
    "Dogecoin": "DOGE-USD",
    "Polygon": "MATIC-USD",
    "Polkadot": "DOT-USD",
}
st.session_state["assets"]["Stocks & ETFs"] = stocks_and_etfs
st.session_state["assets"]["Forex"] = forex
st.session_state["assets"]["Crypto"] = crypto


if data_fetch_way == "Fetch over the internet":
    if "all_areas_filled" not in st.session_state:
        st.session_state["all_areas_filled"] = False
    if "fetch_data_button_clicked" not in st.session_state:
        st.session_state["fetch_data_button_clicked"] = False
    if "show_data_button_clicked" not in st.session_state:
        st.session_state["show_data_button_clicked"] = False
    if "show_chart_button_clicked" not in st.session_state:
        st.session_state["show_chart_button_clicked"] = False
    if "chart_data_selectbox_clicked" not in st.session_state:
        st.session_state["chart_data_selectbox_clicked"] = False

    market = col2.selectbox(
        "Select the market: ",
        ["<Select>", "Stocks & ETFs", "Forex", "Crypto"],
        on_change=clear_data,
    )
    if market != "<Select>":
        assets = list(st.session_state["assets"][market].keys())
        assets.insert(0, "<Select>")
        asset = col2.selectbox(
            "Select the asset: ", assets, on_change=clear_data
        )
        intervals = ["1m", "1d", "5d", "1wk", "1mo", "3mo"]
        intervals.insert(0, "<Select>")
        interval = col2.selectbox(
            "Select the time frame: ", intervals, on_change=clear_data
        )
        if asset != "<Select>" and interval != "<Select>":
            tickers = st.session_state["assets"][market][asset]
            st.session_state["ticker"] = tickers
            full_data = yf.download(tickers=tickers, interval=interval)
            start = full_data.index[0]
            end = full_data.index[-1]
            val1 = full_data.index[(len(full_data) // 3)]
            val2 = full_data.index[(len(full_data) * 2 // 3)]
            if interval in ["1d", "5d", "1wk", "1mo", "3mo"]:
                start, end = st.select_slider(
                    "Please select the start and end dates:",
                    options=full_data.index,
                    value=(val1, val2),
                    format_func=lambda date: date.strftime("%d-%m-%Y"),
                    on_change=clear_data,
                )
            else:
                start, end = st.select_slider(
                    "Please select the start and end dates:",
                    options=full_data.index,
                    value=(val1, val2),
                    format_func=lambda date: date.strftime(
                        "%d-%m-%Y %H:%M:%S"
                    ),
                    on_change=clear_data,
                )
            adjust_situation = col2.selectbox(
                "Do you want to adjust the prices: ", ["<Select>", "Yes", "No"]
            )
            auto_adjust = adjust_situation == "Yes"

            st.session_state["all_areas_filled"] = (
                market != "<Select>"
                and start != "Type Here ..."
                and end != "Type Here ..."
                and interval != "<Select>"
                and adjust_situation != "<Select>"
            )

    if (
        st.button("Fetch the Data", on_click=fetch_data_button_click)
        or st.session_state["fetch_data_button_clicked"]
    ):
        if st.session_state["all_areas_filled"] == False:
            st.error("Please fill all the areas.")
        else:
            ohlcv = st.session_state["ohlcv"]
            if ohlcv is None and st.session_state.conf_change == False:
                st.error("Data could not be downloaded.")
            elif ohlcv is not None:
                st.success("Data fetched successfully")
                st.markdown("<br>", unsafe_allow_html=True)
elif data_fetch_way == "Read from a file":
    uploaded_file = col2.file_uploader(
        "Choose a csv or excel file which first word of its name equal to the ticker name to upload "
    )
    st.markdown("<br>", unsafe_allow_html=True)
    if uploaded_file is not None:
        tickers = re.findall(r"[\w']+", uploaded_file.name)[0]
        st.session_state["ticker"] = tickers
        try:
            if uploaded_file.name.endswith(".csv"):
                st.session_state["ohlcv"] = pd.read_csv(
                    uploaded_file, index_col="Date"
                )
            elif uploaded_file.name.endswith(
                ".xlx"
            ) or uploaded_file.name.endswith(".xlsx"):
                st.session_state["ohlcv"] = pd.read_excel(
                    uploaded_file, index_col="Date"
                )
        except Exception:
            st.error("you need to upload a csv or excel file.")
        else:
            st.session_state["ohlcv"].index = pd.to_datetime(
                st.session_state["ohlcv"].index
            )
            ohlcv = st.session_state["ohlcv"]
            st.success("Data fetched successfully")
            st.markdown("<br>", unsafe_allow_html=True)
if st.session_state["ohlcv"] is not None:
    # if st.checkbox("Do you want to smooth the signal?"):
    #    smooth_method = st.selectbox(
    #        "Which way do you want to smooth the signal?",
    #        [
    #            "<Select>",
    #            "Moving Average",
    #            "Heikin-Ashi",
    #            "Trend Normalization",
    #        ],
    #    )
    # if smooth_method != "<Select>":
    #   ohlcv = cd.signal_smoothing(
    #        data=ohlcv,
    #        smoothing_method=smooth_method,
    #        parameters={"window": 20},
    #   )
    #    st.session_state["ohlcv"] = ohlcv
    st.markdown("<br>", unsafe_allow_html=True)

    center_tabular_button = st.button("Show the data in a tabular format")
    if (
        center_tabular_button
        or st.session_state["show_data_button_clicked"] == True
    ):
        st.dataframe(ohlcv, width=1100)
        st.session_state["show_data_button_clicked"] = True

    st.markdown("<br> <br>", unsafe_allow_html=True)

    center_chart_button = st.button("Show the data in a chart")
    if (
        center_chart_button
        or st.session_state["show_chart_button_clicked"] == True
    ):
        st.session_state["show_chart_button_clicked"] = True
        col1, col2, col3 = st.columns([1, 2, 1])
        display_format = col2.selectbox(
            "Select the price to show in the chart: ",
            ["<Select>", "All", "Open", "High", "Low", "Close"],
            on_change=chart_data_selectbox_click,
        )
        if (
            display_format
            != "<Select>"
            # and st.session_state["chart_data_selectbox_clicked"]
        ):
            cd.show_prices(
                data=st.session_state["ohlcv"],
                ticker=tickers,
                show_which_price=display_format,
            )
    st.markdown("<br>", unsafe_allow_html=True)
if (
    data_fetch_way != "<Select>"
    and st.session_state["ohlcv"] is not None
    and st.session_state["ticker"] != "Type Here ..."
    and smooth_method != "<Select>"
):
    col1, col2 = st.columns([1, 1])
    with col1:
        if smooth_method == "None":
            file_name = f"{st.session_state['ticker']}-ohlcv.csv"
        else:
            file_name = (
                f"{st.session_state['ticker']}-ohlcv-{smooth_method}.csv"
            )
        if st.download_button(
            label="Download data as CSV",
            data=st.session_state["ohlcv"].to_csv().encode("utf-8"),
            file_name=file_name,
            mime="text/csv",
        ):
            st.success("Data was saved successfully")
    with col2:
        pass

st.markdown("<br> <br> <br>", unsafe_allow_html=True)
