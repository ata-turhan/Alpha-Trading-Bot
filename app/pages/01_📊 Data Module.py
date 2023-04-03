import base64
import io
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


def fetch_data_button_click(
    tickers, start, end, interval, auto_adjust, fundamentals
) -> None:
    if st.session_state["all_areas_filled"]:
        st.session_state["fetch_data_button_clicked"] = True
        st.session_state["show_data_button_clicked"] = False
        st.session_state["show_chart_button_clicked"] = False
        st.session_state["chart_data_selectbox_clicked"] = False
        data = cd.get_financial_data(
            tickers=tickers,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
        )
    data = cd.fetch_fundamental_data(data, start, end)
    chosen_columns = ["Open", "High", "Low", "Close", "Volume"]
    chosen_columns.extend(fundamentals)
    st.session_state["data"] = data[chosen_columns]


def smooth_data_button_click():
    st.session_state["smooth_data_button_clicked"] = True


def smooth_data_selectbox_click():
    st.session_state["smooth_data_selectbox_clicked"] = True


def show_data_button_click():
    st.session_state["show_data_button_clicked"] = True


def show_chart_button_click():
    st.session_state["show_chart_button_clicked"] = True


def chart_data_selectbox_click():
    st.session_state["chart_data_selectbox_clicked"] = True


def clear_data():
    st.session_state["data"] = None
    st.session_state.conf_change = True
    st.session_state["smooth_data_button_clicked"] = False
    st.session_state["show_data_button_clicked"] = False
    st.session_state["show_chart_button_clicked"] = False
    st.session_state["chart_data_selectbox_clicked"] = False


def main():
    if "conf_change" not in st.session_state:
        st.session_state.conf_change = False
    if "data" not in st.session_state:
        st.session_state["data"] = None
    if "ticker" not in st.session_state:
        st.session_state["ticker"] = ""
    if "assets" not in st.session_state:
        st.session_state["assets"] = {}
    if "smooth_data_button_clicked" not in st.session_state:
        st.session_state["smooth_data_button_clicked"] = False
    if "show_data_button_clicked" not in st.session_state:
        st.session_state["show_data_button_clicked"] = False
    if "show_chart_button_clicked" not in st.session_state:
        st.session_state["show_chart_button_clicked"] = False
    if "chart_data_selectbox_clicked" not in st.session_state:
        st.session_state["chart_data_selectbox_clicked"] = False
    if "data_to_show" not in st.session_state:
        st.session_state["data_to_show"] = None
    if "start" not in st.session_state:
        st.session_state["start"] = None
    if "end" not in st.session_state:
        st.session_state["end"] = None
    if "interval" not in st.session_state:
        st.session_state["interval"] = None
    if "auto_adjust" not in st.session_state:
        st.session_state["auto_adjust"] = None
    if "fundamentals" not in st.session_state:
        st.session_state["fundamentals"] = None

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
    st.set_page_config(page_title="Trading Bot", page_icon="🤖", layout="wide")

    add_bg_from_local("data/background.png", "data/bot.png")

    for _ in range(18):
        st.sidebar.text("\n")
    st.sidebar.write("Developed by Ata Turhan")
    st.sidebar.write("Contact at ataturhan21@gmail.com")

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

    smooth_method = "<Select>"

    if data_fetch_way == "Fetch over the internet":
        if "all_areas_filled" not in st.session_state:
            st.session_state["all_areas_filled"] = False
        if "fetch_data_button_clicked" not in st.session_state:
            st.session_state["fetch_data_button_clicked"] = False

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
            st.session_state["interval"] = interval
            if asset != "<Select>" and interval != "<Select>":
                tickers = st.session_state["assets"][market][asset]
                st.session_state["ticker"] = tickers
                full_data = yf.download(tickers=tickers, interval=interval)
                start = full_data.index[0]
                end = full_data.index[-1]
                val1 = full_data.index[(len(full_data) // 3)]
                val2 = full_data.index[(len(full_data) * 2 // 3)]
                if interval in ["1d", "5d", "1wk", "1mo", "3mo"]:
                    (
                        st.session_state["start"],
                        st.session_state["end"],
                    ) = st.select_slider(
                        "Please select the start and end dates:",
                        options=full_data.index,
                        value=(val1, val2),
                        format_func=lambda date: date.strftime("%d-%m-%Y"),
                        on_change=clear_data,
                    )
                else:
                    (
                        st.session_state["start"],
                        st.session_state["end"],
                    ) = st.select_slider(
                        "Please select the start and end dates:",
                        options=full_data.index,
                        value=(val1, val2),
                        format_func=lambda date: date.strftime(
                            "%d-%m-%Y %H:%M:%S"
                        ),
                        on_change=clear_data,
                    )
                adjust_situation = col2.selectbox(
                    "Do you want to adjust the prices: ",
                    ["<Select>", "Yes", "No"],
                )
                st.session_state["auto_adjust"] = adjust_situation == "Yes"
                st.session_state["fundamentals"] = col2.multiselect(
                    "Besides the price data, which fundamental data do you want to add?",
                    [
                        "FED 2Y Interest Rate",
                        "FED 10Y Interest Rate",
                        "Yield Difference",
                        "CPI",
                    ],
                )
                st.session_state["all_areas_filled"] = (
                    market != "<Select>"
                    and start != "Type Here ..."
                    and end != "Type Here ..."
                    and interval != "<Select>"
                    and adjust_situation != "<Select>"
                )

        if (
            st.button(
                "Fetch the data",
                on_click=fetch_data_button_click,
                args=(
                    st.session_state["ticker"],
                    st.session_state["start"],
                    st.session_state["end"],
                    st.session_state["interval"],
                    st.session_state["auto_adjust"],
                    st.session_state["fundamentals"],
                ),
            )
            or st.session_state["fetch_data_button_clicked"]
        ):
            if st.session_state["all_areas_filled"] == False:
                st.error("Please fill all the areas.")
            else:
                data = st.session_state["data"]
                if data is None and st.session_state.conf_change == False:
                    st.error("Data could not be downloaded.")
                elif data is not None:
                    st.success("Data fetched successfully")
                    st.markdown("<br>", unsafe_allow_html=True)
    elif data_fetch_way == "Read from a file":
        col2.markdown("<br><br>", unsafe_allow_html=True)
        uploaded_file = col2.file_uploader(
            "To upload, select a csv or excel file with the first word of its name matching the ticker name.",
            on_change=clear_data,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if uploaded_file is not None:
            tickers = re.findall(r"[\w']+", uploaded_file.name)[0]
            st.session_state["ticker"] = tickers
            try:
                if uploaded_file.name.endswith(".csv"):
                    st.session_state["data"] = pd.read_csv(
                        uploaded_file, index_col="Date"
                    )
                elif uploaded_file.name.endswith(
                    ".xlx"
                ) or uploaded_file.name.endswith(".xlsx"):
                    st.session_state["data"] = pd.read_excel(
                        uploaded_file, index_col="Date"
                    )
            except IOError:
                st.error("You need to upload a csv or excel file.")
            except Exception:
                st.error("An unknown error occurred.")
            else:
                st.session_state["data"].index = pd.to_datetime(
                    st.session_state["data"].index
                )
                data = st.session_state["data"]
                st.success("Data fetched successfully")
                st.markdown("<br>", unsafe_allow_html=True)
    if st.session_state["data"] is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.session_state["data_to_show"] = st.session_state["data"].copy()

        smooth_button = st.button("Smooth the data")
        if (
            smooth_button
            or st.session_state["smooth_data_button_clicked"] == True
        ):
            st.session_state["smooth_data_button_clicked"] = True
            col1, col2, col3 = st.columns([1, 2, 1])
            smooth_method = col2.selectbox(
                "Select the method to smooth the data: ",
                [
                    "<Select>",
                    "None",
                    "Moving Average",
                    "Heikin Ashi",
                    "Trend Normalization",
                ],
                on_change=smooth_data_selectbox_click,
            )
            if smooth_method != "<Select>":
                st.session_state["data_to_show"] = cd.signal_smoothing(
                    df=st.session_state["data"],
                    smoothing_method=smooth_method,
                    parameters={"window": 20},
                )
        st.markdown("<br> <br>", unsafe_allow_html=True)
        center_tabular_button = st.button("Show the data in a tabular format")
        if (
            center_tabular_button
            or st.session_state["show_data_button_clicked"] == True
        ):
            st.dataframe(st.session_state["data_to_show"], width=1100)
            st.session_state["show_data_button_clicked"] = True

        st.markdown("<br> <br>", unsafe_allow_html=True)

        center_chart_button = st.button("Show the data in a chart")
        if (
            center_chart_button
            or st.session_state["show_chart_button_clicked"]
        ):
            st.session_state["show_chart_button_clicked"] = True
            col1, col2, col3 = st.columns([1, 2, 1])
            display_format = col2.selectbox(
                "Select the price to show in the chart: ",
                ["<Select>", "All", "Open", "High", "Low", "Close"],
                on_change=chart_data_selectbox_click,
            )
            if (
                display_format != "<Select>"
                and st.session_state["chart_data_selectbox_clicked"]
            ):
                cd.show_prices(
                    data=st.session_state["data_to_show"],
                    ticker=tickers,
                    show_which_price=display_format,
                )
        st.markdown("<br><br>", unsafe_allow_html=True)
    if data_fetch_way != "<Select>" and st.session_state["data"] is not None:
        col1, col2, col3, col4, col5, col6, col7 = st.columns(
            [1, 2, 1, 1, 1, 2, 1]
        )
        if smooth_method in ["<Select>", "None"]:
            file_name = f"{st.session_state['ticker']}-data"
        else:
            file_name = f"{st.session_state['ticker']}-data-{smooth_method}"
        if col2.download_button(
            label="Download data as CSV",
            data=st.session_state["data"].to_csv().encode("utf-8"),
            file_name=file_name + ".csv",
            mime="text/csv",
        ):
            col2.success("Data was saved successfully")
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            st.session_state["data"].to_excel(writer, sheet_name="Data")
            writer.save()
            if col6.download_button(
                label="Download data as Excel",
                data=buffer,
                file_name=file_name + ".xlsx",
                mime="application/vnd.ms-excel",
            ):
                col6.success("Data was saved successfully")


if __name__ == "__main__":
    main()
