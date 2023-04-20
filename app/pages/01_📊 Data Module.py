import re
from io import BytesIO

import pandas as pd
import streamlit as st
import yfinance as yf
from configuration import add_bg_from_local, configure_authors, configure_page
from create_data import (
    create_technical_indicators,
    fetch_fundamental_data,
    fred_codes,
    get_financial_data,
    show_prices,
    signal_smoothing,
)


def set_session_variables():
    if "conf_change" not in st.session_state:
        st.session_state.conf_change = True
    if "ticker" not in st.session_state:
        st.session_state["ticker"] = ""
    if "indicators" not in st.session_state:
        st.session_state["indicators"] = None
    if "fundamentals" not in st.session_state:
        st.session_state["fundamentals"] = None
    if "assets" not in st.session_state:
        st.session_state["assets"] = None
    if "data" not in st.session_state:
        st.session_state["data"] = None
    if "data_to_show" not in st.session_state:
        st.session_state["data_to_show"] = None
    if "all_areas_filled" not in st.session_state:
        st.session_state["all_areas_filled"] = False
    if "fetch_data_button_clicked" not in st.session_state:
        st.session_state["fetch_data_button_clicked"] = False
    if "smooth_data_button_clicked" not in st.session_state:
        st.session_state["smooth_data_button_clicked"] = False
    if "smooth_data_selectbox_clicked" not in st.session_state:
        st.session_state["smooth_data_selectbox_clicked"] = False
    if "show_data_button_clicked" not in st.session_state:
        st.session_state["show_data_button_clicked"] = False
    if "show_chart_button_clicked" not in st.session_state:
        st.session_state["show_chart_button_clicked"] = False
    if "show_chart_selectbox_clicked" not in st.session_state:
        st.session_state["show_chart_selectbox_clicked"] = False


def fetch_data_button_click(
    tickers, start, end, interval, auto_adjust, col2
) -> None:
    if (
        st.session_state["all_areas_filled"]
        and st.session_state.conf_change == True
    ):
        st.session_state["fetch_data_button_clicked"] = True
        reset_click()
        _, col2, _ = st.columns([1, 2, 1])
        with col2:
            with st.spinner("Fetching ohlcv data from Yahoo Finance"):
                data = get_financial_data(
                    tickers=tickers,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=auto_adjust,
                )
                st.session_state["data"] = data
                st.session_state.conf_change = False
    else:
        col2.error("Please fill all the areas.")


# a method to make all button and selectbox click boolean values false
def reset_click():
    st.session_state["smooth_data_button_clicked"] = False
    st.session_state["smooth_data_selectbox_clicked"] = False
    st.session_state["show_data_button_clicked"] = False
    st.session_state["show_chart_button_clicked"] = False
    st.session_state["show_chart_selectbox_clicked"] = False


def add_fundamental_data(fundamentals):
    data = st.session_state["data"]
    start = data.index[0]
    end = data.index[-1]
    if "Gross Domestic Product" not in list(data.columns):
        _, col2, _ = st.columns([1, 2, 1])
        with col2:
            with st.spinner("Fetching fundamental data"):
                data = fetch_fundamental_data(data, start, end)
                st.session_state["data"] = data
    columns = ["Open", "High", "Low", "Close", "Volume"]
    if len(fundamentals) > 0:
        columns.extend(fundamentals)
    st.session_state["data_to_show"] = data[columns]


def add_indicator_data(indicators):
    columns = list(st.session_state["data_to_show"].columns)
    if len(indicators) > 0:
        columns.extend(indicators)
    st.session_state["data_to_show"] = st.session_state["data"][columns]


def smooth_data_button_click():
    st.session_state["smooth_data_button_clicked"] = True


def smooth_data_selectbox_click():
    st.session_state["smooth_data_selectbox_clicked"] = True


def show_data_button_click():
    st.session_state["show_data_button_clicked"] = True


def show_chart_button_click():
    st.session_state["show_chart_button_clicked"] = True


def chart_data_selectbox_click():
    st.session_state["show_chart_selectbox_clicked"] = True


def clear_data():
    st.session_state["data"] = None
    st.session_state["data_to_show"] = None
    st.session_state.conf_change = True
    reset_click()


def load_tickers():
    _, col2, _ = st.columns(3)
    with col2:
        with st.spinner("Getting Tickers..."):
            tickers_dict = {}
            markets = ["Stock", "ETF", "Forex", "Crypto"]
            tickers = pd.read_excel(
                "data/tickers.xlsx",
                sheet_name=None,
            )
            for market in markets:
                market_df = tickers[f"{market}"].set_index("Symbol")
                market_dict = market_df["Name"].to_dict()
                tickers_dict[market] = market_dict
            tickers_list = {}
            for market in markets:
                tickers_list[market] = [
                    f"{k} - {v}" for k, v in tickers_dict[market].items()
                ]
    return tickers_dict, tickers_list


def get_tickers():
    if (
        "tickers_dict" not in st.session_state
        and "tickers_list" not in st.session_state
    ):
        (
            st.session_state["tickers_dict"],
            st.session_state["tickers_list"],
        ) = load_tickers()


def main():
    set_session_variables()
    configure_page()
    configure_authors()
    add_bg_from_local("data/background.png", "data/bot.png")

    DEFAULT_CHOICE = "<Select>"

    st.markdown(
        "<h1 style='text-align: center; color: black; font-size: 65px;'> 📊 Data Module </h1> <br> ",
        unsafe_allow_html=True,
    )

    style = "<style>.row-widget.stButton {text-align: center;}</style>"
    st.markdown(style, unsafe_allow_html=True)

    _, col2, _ = st.columns([1, 2, 1])
    data_fetch_way = col2.selectbox(
        "Which way do you want to get the prices: ",
        [DEFAULT_CHOICE, "Fetch over the internet", "Read from a file"],
        on_change=clear_data,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    smooth_method = DEFAULT_CHOICE

    if data_fetch_way == "Fetch over the internet":
        get_tickers()
        start, end, interval, auto_adjust = [None] * 4
        market = col2.selectbox(
            "Select the market: ",
            [DEFAULT_CHOICE, "Stock", "ETF", "Forex", "Crypto"],
            on_change=clear_data,
        )
        if market != DEFAULT_CHOICE:
            assets = st.session_state["tickers_list"][market]
            assets = [DEFAULT_CHOICE] + assets
            asset = col2.selectbox(
                "Select the asset: ", assets, on_change=clear_data
            )
            if asset != DEFAULT_CHOICE:
                asset = asset.split("-")[0].strip()
            intervals = [
                "1m",
                "1d",
                "5d",
                "1wk",
                "1mo",
                "3mo",
            ]
            intervals.insert(0, DEFAULT_CHOICE)
            interval = col2.selectbox(
                "Select the time frame: ", intervals, on_change=clear_data
            )
            if asset != DEFAULT_CHOICE and interval != DEFAULT_CHOICE:
                ticker = asset
                st.write(ticker)
                st.write(len(ticker))
                if ticker is None:
                    full_data = None
                else:
                    st.session_state["ticker"] = ticker
                    full_data = yf.download(tickers=ticker, interval=interval)
                if full_data is None or len(full_data) == 0:
                    col2.error(
                        "Yahoo do not have this ticker data. Please try another ticker."
                    )
                else:
                    val1 = full_data.index[(len(full_data) // 3)]
                    val2 = full_data.index[(len(full_data) * 2 // 3)]
                    if interval in ["1d", "5d", "1wk", "1mo", "3mo"]:
                        (start, end) = st.select_slider(
                            "Please select the start and end dates:",
                            options=full_data.index,
                            value=(val1, val2),
                            format_func=lambda date: date.strftime("%d-%m-%Y"),
                            on_change=clear_data,
                        )
                    else:
                        (start, end) = st.select_slider(
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
                [DEFAULT_CHOICE, "Yes", "No"],
            )

            st.session_state["all_areas_filled"] = (
                market != DEFAULT_CHOICE
                and start != None
                and end != None
                and interval != DEFAULT_CHOICE
                and adjust_situation != DEFAULT_CHOICE
            )
        if (
            st.button(
                "Fetch the data",
                on_click=fetch_data_button_click,
                args=(
                    st.session_state["ticker"],
                    start,
                    end,
                    interval,
                    auto_adjust,
                    col2,
                ),
            )
            or st.session_state["fetch_data_button_clicked"]
        ):
            if st.session_state["all_areas_filled"]:
                data = st.session_state["data"]
                if data is None and st.session_state.conf_change == False:
                    st.error("Data could not be downloaded.")
                elif data is not None:
                    col2.success("Data fetched successfully")
                    st.markdown("<br>", unsafe_allow_html=True)
    elif data_fetch_way == "Read from a file" and st.session_state.conf_change:
        col2.markdown("<br><br>", unsafe_allow_html=True)
        uploaded_file = col2.file_uploader(
            "To upload, select a csv or excel file whose first word matches the ticker name.",
            on_change=clear_data,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if uploaded_file is not None:
            ticker = re.findall(r"[\w']+", uploaded_file.name)[0]
            st.session_state["ticker"] = ticker
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
                col2.error("You need to upload a csv or excel file.")
            except Exception:
                col2.error("An unknown error occurred.")
            else:
                st.session_state["data"].index = pd.to_datetime(
                    st.session_state["data"].index
                )
                data = st.session_state["data"]
                col2.success("Data fetched successfully")
                st.session_state.conf_change = False
    if st.session_state["data"] is not None:
        _, col2, _ = st.columns([1, 2, 1])
        st.session_state["fundamentals"] = col2.multiselect(
            "Besides the price data, which fundamental data do you want to add?",
            fred_codes.keys(),
        )
        add_fundamental_data(st.session_state["fundamentals"])

        if "ti_momentum_rsi" not in list(st.session_state["data"].columns):
            st.session_state["data"] = create_technical_indicators(
                st.session_state["data"]
            )
        st.session_state["indicators"] = col2.multiselect(
            "Besides the price data, which technical indicators data do you want to add?",
            [col for col in st.session_state["data"].columns if "ti_" in col],
        )
        add_indicator_data(st.session_state["indicators"])

        st.markdown("<br><br>", unsafe_allow_html=True)
        smooth_button = st.button("Smooth the data")
        if smooth_button or st.session_state["smooth_data_button_clicked"]:
            st.session_state["smooth_data_button_clicked"] = True
            _, col2, _ = st.columns([1, 2, 1])
            smooth_method = col2.selectbox(
                "Select the method to smooth the data: ",
                [
                    DEFAULT_CHOICE,
                    "None",
                    "Moving Average",
                    "Heikin-Ashi",
                    "Trend Normalization",
                ],
                on_change=smooth_data_selectbox_click,
            )
            if smooth_method != DEFAULT_CHOICE:
                st.session_state["data_to_show"] = signal_smoothing(
                    df=st.session_state["data"],
                    smoothing_method=smooth_method,
                    parameters={"window": 20},
                )
        st.markdown("<br> <br>", unsafe_allow_html=True)
        show_data_button = st.button("Show the data in a tabular format")
        if (
            show_data_button
            or st.session_state["show_data_button_clicked"] == True
        ):
            st.dataframe(st.session_state["data_to_show"], width=1100)
            st.session_state["show_data_button_clicked"] = True

        st.markdown("<br> <br>", unsafe_allow_html=True)

        show_chart_button = st.button("Show the data in a chart")
        if show_chart_button or st.session_state["show_chart_button_clicked"]:
            st.session_state["show_chart_button_clicked"] = True
            _, col2, _ = st.columns([1, 2, 1])
            columns_to_chart = [
                DEFAULT_CHOICE,
                "Candlestick",
                "Open",
                "High",
                "Low",
                "Close",
            ]
            columns_to_chart.extend(st.session_state["fundamentals"])
            columns_to_chart.extend(st.session_state["indicators"])
            display_format = col2.multiselect(
                "Select the data to show in the chart: ",
                columns_to_chart,
                on_change=chart_data_selectbox_click,
            )
            if (
                len(display_format) != 0
                and st.session_state["show_chart_selectbox_clicked"]
            ):
                show_prices(
                    data=st.session_state["data_to_show"],
                    ticker=st.session_state["ticker"],
                    show_which_price=display_format,
                )
        st.markdown("<br><br>", unsafe_allow_html=True)
    if (
        data_fetch_way != DEFAULT_CHOICE
        and st.session_state["data"] is not None
    ):
        _, col2, _, _, _, col6, _ = st.columns([1, 2, 1, 1, 1, 2, 1])
        if smooth_method in [DEFAULT_CHOICE, "None"]:
            file_name = f"{st.session_state['ticker']}-data"
        else:
            file_name = f"{st.session_state['ticker']}-data-{smooth_method}"
        if col2.download_button(
            label="Download data as CSV",
            data=st.session_state["data_to_show"].to_csv().encode("utf-8"),
            file_name=file_name + ".csv",
            mime="text/csv",
        ):
            col2.success("Data was saved successfully")
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            st.session_state["data_to_show"].to_excel(
                writer, sheet_name="Data"
            )
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
