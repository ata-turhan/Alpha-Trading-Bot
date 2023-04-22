import create_data as cd
import create_strategy as cs
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from configuration import add_bg_from_local, configure_authors, configure_page
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)


def set_session_variables():
    if "data" not in st.session_state:
        st.session_state.data = None
    if "smoothed_data" not in st.session_state:
        st.session_state.smoothed_data = None
    if "ai_data" not in st.session_state:
        st.session_state.ai_data = None
    if "strategies" not in st.session_state:
        st.session_state.strategies = {}
    if "signals" not in st.session_state:
        st.session_state.signals = None
    if "ticker" not in st.session_state:
        st.session_state.ticker = ""
    if "strategy_keys" not in st.session_state:
        st.session_state.strategy_keys = set()
    if "last_strategy" not in st.session_state:
        st.session_state.last_strategy = ""
    if "all_strategy_areas_filled" not in st.session_state:
        st.session_state.all_strategy_areas_filled = False


def clean_signals():
    st.session_state.signals = None


def main():
    set_session_variables()
    configure_page()
    configure_authors()
    add_bg_from_local("data/background.png", "data/bot.png")
    # st.dataframe(st.session_state["smoothed_data"])

    DEFAULT_CHOICE = "<Select>"

    st.markdown(
        "<h1 style='text-align: center; color: black; font-size: 65px;'> 💡 Strategy Module </h1> <br> <br>",
        unsafe_allow_html=True,
    )
    _, center_col_get, _ = st.columns([1, 2, 1])

    style = "<style>.row-widget.stButton {text-align: center;}</style>"
    st.markdown(style, unsafe_allow_html=True)

    if st.session_state["data"] is None:
        st.error("Please get the data first.")
    else:
        key_name = ""
        strategy_fetch_way = center_col_get.selectbox(
            "Which way do you want to get the signals of a strategy: ",
            [DEFAULT_CHOICE, "Create a strategy", "Read from a file"],
            on_change=clean_signals,
        )
        # st.markdown("<br> <br>", unsafe_allow_html=True)

        if strategy_fetch_way == "Read from a file":
            uploaded_file = center_col_get.file_uploader(
                "Choose a csv file to upload"
            )
            if uploaded_file is not None:
                try:
                    st.session_state["signals"] = pd.read_csv(
                        uploaded_file,
                        index_col="Date",
                    )
                except FileNotFoundError as exception:
                    center_col_get.error(
                        "you need to upload a csv or excel file."
                    )
                except Except as e:
                    center_col_get.error(e)
                else:
                    signals = st.session_state["signals"]
                    if signals is not None:
                        st.session_state["smoothed_data"] = st.session_state[
                            "data"
                        ]
                        key_name = f"Uploaded Signals ({uploaded_file.name.replace('.csv', '')})"
                        st.session_state.last_strategy = key_name
                        if key_name not in st.session_state.strategy_keys:
                            st.session_state.strategy_keys.add(key_name)
                            key = f"S{len(st.session_state['strategies'])+1} - {key_name}"
                            st.session_state["strategies"][
                                key
                            ] = st.session_state["signals"]

                            # st.markdown("<br>", unsafe_allow_html=True)
                            center_col_get.success(
                                "The signals of the strategy fetched successfully"
                            )
                            # st.write(st.session_state["strategies"])
        elif strategy_fetch_way == "Create a strategy":
            smooth_method = center_col_get.selectbox(
                "Which way do you want to use the price data with smoothing?",
                [
                    "None",
                    "Moving Average",
                    "Heikin-Ashi",
                    "Trend Normalization",
                ],
                on_change=clean_signals,
            )
            st.session_state["smoothed_data"] = cd.signal_smoothing(
                df=st.session_state["data"],
                smoothing_method=smooth_method,
                parameters={"window": 20},
            )
            func = ""
            params = {}
            strategy_type = center_col_get.selectbox(
                "Which strategy do you want to create: ",
                [
                    DEFAULT_CHOICE,
                    "Indicator Trading",
                    "Momentum Trading",
                    "Candlestick Pattern Trading",
                    "Candlestick Sentiment Trading",
                    "Support-Resistance Trading",
                    "AI Trading",
                ],
                on_change=clean_signals,
            )
            if strategy_type == "Indicator Trading":
                indicator = center_col_get.selectbox(
                    "Select the indicator you want to use: ",
                    [DEFAULT_CHOICE, "RSI", "SMA", "EMA", "Bollinger Bands"],
                    on_change=clean_signals,
                )
                if indicator != DEFAULT_CHOICE:
                    if indicator == "RSI":
                        col1, col2 = center_col_get.columns(2, gap="large")
                        with col1:
                            oversold = st.number_input(
                                "Please enter the oversold value",
                                value=30,
                                on_change=clean_signals,
                            )
                        with col2:
                            overbought = st.number_input(
                                "Please enter the overbought value",
                                value=70,
                                on_change=clean_signals,
                            )
                        func = cs.rsi_trading
                        params = {
                            "ohlcv": st.session_state["smoothed_data"],
                            "oversold": oversold,
                            "overbought": overbought,
                        }
                        key_name = f"{strategy_type} - {indicator}  ({oversold}, {overbought})"
                    if indicator == "SMA":
                        col1, col2 = center_col_get.columns(2, gap="medium")
                        with col1:
                            short_smo = st.number_input(
                                "Please enter the short moving average value",
                                value=50,
                                on_change=clean_signals,
                            )
                        with col2:
                            long_smo = st.number_input(
                                "Please enter the long moving average value",
                                value=200,
                                on_change=clean_signals,
                            )
                        func = cs.sma_trading
                        params = {
                            "ohlcv": st.session_state["smoothed_data"],
                            "short_smo": short_smo,
                            "long_smo": long_smo,
                        }
                        key_name = f"{strategy_type} - {indicator}  ({short_smo}, {long_smo})"
                    if indicator == "EMA":
                        col1, col2 = center_col_get.columns(2, gap="medium")
                        with col1:
                            short_emo = st.number_input(
                                "Please enter the short moving average value",
                                value=50,
                                on_change=clean_signals,
                            )
                        with col2:
                            long_emo = st.number_input(
                                "Please enter the long moving average value",
                                value=200,
                                on_change=clean_signals,
                            )
                        func = cs.ema_trading
                        params = {
                            "ohlcv": st.session_state["smoothed_data"],
                            "short_emo": short_emo,
                            "long_emo": long_emo,
                        }
                        key_name = f"{strategy_type} - {indicator}  ({short_emo}, {long_emo})"
                    if indicator == "Bollinger Bands":
                        col1, col2 = center_col_get.columns(2, gap="small")
                        with col1:
                            window = st.number_input(
                                "Please enter the window value",
                                value=20,
                                on_change=clean_signals,
                            )
                        with col2:
                            window_dev = st.number_input(
                                "Please enter the window deviation value",
                                value=2,
                                on_change=clean_signals,
                            )
                        func = cs.bb_trading
                        params = {
                            "ohlcv": st.session_state["smoothed_data"],
                            "window": window,
                            "window_dev": window_dev,
                        }
                        key_name = f"{strategy_type} - {indicator}  ({window}, {window_dev})"
            elif strategy_type == "Momentum Trading":
                variation = center_col_get.selectbox(
                    "Select the momentum strategy you want to use: ",
                    [
                        DEFAULT_CHOICE,
                        "Momentum Day Trading",
                        "Momentum Percentage Trading",
                    ],
                    on_change=clean_signals,
                )
                if variation != DEFAULT_CHOICE:
                    if variation == "Momentum Day Trading":
                        col1, col2 = center_col_get.columns(2)
                        with col1:
                            up_day = st.number_input(
                                "Please enter the up day number",
                                value=3,
                                on_change=clean_signals,
                            )
                        with col2:
                            down_day = st.number_input(
                                "Please enter the down day number",
                                value=3,
                                on_change=clean_signals,
                            )
                        _, col2, _ = center_col_get.columns([1, 3, 1])
                        with col2:
                            reverse = st.checkbox(
                                "Reverse the logic of the strategy",
                                on_change=clean_signals,
                            )
                        func = cs.momentum_day_trading
                        params = {
                            "ohlcv": st.session_state["smoothed_data"],
                            "up_day": up_day,
                            "down_day": down_day,
                            "reverse": reverse,
                        }
                    if variation == "Momentum Percentage Trading":
                        col1, col2 = center_col_get.columns(2, gap="medium")
                        with col1:
                            up_day = st.number_input(
                                "Please enter the up day number",
                                value=3,
                                on_change=clean_signals,
                            )
                        with col2:
                            up_percentage = st.number_input(
                                "Please enter the up percentage number",
                                value=3,
                                on_change=clean_signals,
                            )
                        col1, col2 = center_col_get.columns(
                            [4, 5], gap="medium"
                        )
                        with col1:
                            down_day = st.number_input(
                                "Please enter the down day number",
                                value=3,
                                on_change=clean_signals,
                            )
                        with col2:
                            down_percentage = st.number_input(
                                "Please enter the down percentage number",
                                value=3,
                                on_change=clean_signals,
                            )
                        _, col2, _ = center_col_get.columns([1, 3, 1])
                        with col2:
                            reverse = st.checkbox(
                                "Reverse the logic of the strategy",
                                on_change=clean_signals,
                            )

                        func = cs.momentum_percentage_trading
                        params = {
                            "ohlcv": st.session_state["smoothed_data"],
                            "up_percentage": up_percentage,
                            "up_day": up_day,
                            "down_percentage": down_percentage,
                            "down_day": down_day,
                            "reverse": reverse,
                        }
                    key_name = f"{strategy_type} - {variation}  ({up_day}, {down_day}, reverse={reverse})"
            elif strategy_type == "Candlestick Pattern Trading":
                col1, col2 = st.columns([1, 1])
                with col1:
                    buy_pattern = center_col_get.selectbox(
                        "Select the pattern you want to use for a buy signal:",
                        cs.candlestick_bullish_patterns,
                        on_change=clean_signals,
                    )
                with col2:
                    sell_pattern = center_col_get.selectbox(
                        "Select the pattern you want to use for a sell signal:",
                        cs.candlestick_bearish_patterns,
                        on_change=clean_signals,
                    )
                func = cs.candlestick_pattern_trading
                params = {
                    "ohlcv": st.session_state["smoothed_data"],
                    "buy_pattern": buy_pattern,
                    "sell_pattern": sell_pattern,
                }
                key_name = f"{strategy_type} ({buy_pattern}, {sell_pattern})"
            elif strategy_type == "Candlestick Sentiment Trading":
                col1, col2 = st.columns([1, 1])
                with col1:
                    consecutive_bullish_num = center_col_get.number_input(
                        "Please enter the the consecutive number of \
                        bullish candles to trigger the buy signal",
                        value=5,
                        on_change=clean_signals,
                    )
                with col2:
                    consecutive_bearish_num = center_col_get.number_input(
                        "Please enter the the consecutive number of \
                        bearish candles to trigger the sell signal",
                        value=5,
                        on_change=clean_signals,
                    )
                func = cs.candlestick_sentiment_trading
                params = {
                    "ohlcv": st.session_state["smoothed_data"],
                    "consecutive_bullish_num": consecutive_bullish_num,
                    "consecutive_bearish_num": consecutive_bearish_num,
                }
                key_name = f"{strategy_type} ({consecutive_bullish_num}, {consecutive_bearish_num})"
            elif strategy_type == "Support-Resistance Trading":
                col1, col2 = center_col_get.columns([1, 1])
                with col1:
                    rolling_wave_length = st.number_input(
                        "Please enter the rolling wave length",
                        value=20,
                        on_change=clean_signals,
                    )
                with col2:
                    num_clusters = st.number_input(
                        "Please enter the cluster numbers",
                        value=4,
                        on_change=clean_signals,
                    )
                func = cs.support_resistance_trading
                params = {
                    "ohlcv": st.session_state["smoothed_data"],
                    "rolling_wave_length": rolling_wave_length,
                    "num_clusters": num_clusters,
                }
                key_name = (
                    f"{strategy_type} ({rolling_wave_length}, {num_clusters})"
                )
            elif strategy_type == "AI Trading":
                # st.dataframe(st.session_state.data)
                st.session_state.ai_data = cd.create_labels(
                    ohlcv=st.session_state.data
                )
                # st.dataframe(st.session_state.ai_data)

                # st.dataframe(train)
                # st.dataframe(test)
                ai_method = center_col_get.selectbox(
                    "Select the AI method you want to use: ",
                    [
                        DEFAULT_CHOICE,
                        "Basic Machine Learning",
                        "Advanced Machine Learning",
                        "Deep Learning",
                    ],
                )
                if ai_method == "Basic Machine Learning":
                    bml_model = center_col_get.selectbox(
                        "Select the basic machine learning model you want to use: ",
                        [
                            DEFAULT_CHOICE,
                            "Logistic Regression",
                            "Support Vector Machine",
                            "Random Forest",
                        ],
                        on_change=clean_signals,
                    )
                    if bml_model != DEFAULT_CHOICE:
                        func = cs.basic_ml_trading
                        params = {
                            "bml_model": bml_model,
                            "data": st.session_state.ai_data,
                        }
                        key_name = f"{bml_model}"
                elif ai_method == "Advanced Machine Learning":
                    aml_model = center_col_get.selectbox(
                        "Select the advanced machine learning model you want to use: ",
                        [
                            DEFAULT_CHOICE,
                            "Gradient Boosting Classifier",
                            "XGBoost",
                            "LightGBM",
                        ],
                        on_change=clean_signals,
                    )
                    if aml_model != DEFAULT_CHOICE:
                        func = cs.advanced_ml_trading
                        params = {
                            "aml_model": aml_model,
                            "data": st.session_state.ai_data,
                        }
                        key_name = f"{aml_model}"
                elif ai_method == "Deep Learning":
                    dl_model_layer = center_col_get.number_input(
                        "Please enter the number of layers for your neural network",
                        value=0,
                        on_change=clean_signals,
                    )
                    if dl_model_layer > 0:
                        func = cs.dl_trading
                        params = {
                            "dl_model_layer": dl_model_layer,
                            "data": st.session_state.ai_data,
                        }
                        key_name = f"Neural Network ({dl_model_layer}-Layers)"

            if st.button("Create the signals of the strategy."):
                if strategy_type == "AI Trading":
                    st.session_state["signals"], test_data = func(**params)
                else:
                    st.session_state["signals"] = func(**params)
                if st.session_state["signals"] is not None and key_name != "":
                    st.success(
                        "The signals of the strategy created successfully"
                    )
                    st.session_state.last_strategy = key_name
                    if key_name not in st.session_state.strategy_keys:
                        st.session_state.strategy_keys.add(key_name)
                        key = f"S{len(st.session_state['strategies'])+1} - {key_name}"
                        st.session_state["strategies"][key] = st.session_state[
                            "signals"
                        ]
                    if strategy_type == "Indicator Trading":
                        cs.draw_technical_indicators(
                            ohlcv=st.session_state["smoothed_data"],
                            indicator_name=indicator,
                        )
                        break_line = '<hr style="height:2px;border-width:10;color:black;background-color:black">'
                        st.markdown(break_line, unsafe_allow_html=True)
                    elif strategy_type == "AI Trading":
                        y_test, y_pred = test_data[0], test_data[1]
                        target_names = ["Hold", "Buy", "Sell"]
                        cm = confusion_matrix(
                            y_test,
                            y_pred,
                        )
                        plt.grid(False)
                        disp = ConfusionMatrixDisplay(
                            confusion_matrix=cm,
                            display_labels=target_names,
                        )
                        disp.plot(cmap=plt.cm.Blues)
                        disp.ax_.grid(False)
                        fig = disp.ax_.get_figure()
                        fig.set_size_inches(5, 4)
                        col1, col2 = st.columns([1, 1])
                        col1.subheader("Confusion matrix of the AI Model")
                        col1.pyplot(fig)
                        col2.subheader("Classification report of the AI Model")
                        report = classification_report(
                            y_test,
                            y_pred,
                            target_names=target_names,
                            output_dict=True,
                        )
                        df_report = pd.DataFrame(report).transpose()
                        col2.dataframe(df_report, width=500, height=410)
                        break_line = '<hr style="height:2px;border-width:10;color:black;background-color:black">'
                        st.markdown(break_line, unsafe_allow_html=True)
    st.session_state.all_strategy_areas_filled = (
        key_name != "" and strategy_fetch_way != DEFAULT_CHOICE
    )
    if (
        st.session_state["smoothed_data"] is not None
        and st.session_state["signals"] is not None
        and st.session_state.last_strategy != ""
        and st.session_state.all_strategy_areas_filled
        and len(st.session_state["strategies"]) > 0
    ):
        cs.show_signals_on_chart(
            ohlcv=st.session_state["smoothed_data"],
            signals=st.session_state["signals"],
            last_strategy_name=st.session_state.last_strategy,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        _, center_col_strategies_created, _ = st.columns([1, 2, 1])
        center_col_strategies_created.subheader(
            "These strategies have been created:"
        )
        st.markdown("<br>", unsafe_allow_html=True)

        strategies = st.session_state["strategies"]
        # st.write(strategies)
        for key, val in strategies.items():
            _, strategies_col, _ = st.columns([1, 2, 1])
            strategies_col.write(key)
        st.markdown("<br><br>", unsafe_allow_html=True)
        _, center_col3, _ = st.columns([1, 2, 1])

        mixing_logic = center_col3.text_input(
            "Write your logic to mix the strategies with and & or:",
            help="For example: 'S1 and S2 and S3'",
        )
        if st.button("Mix the strategies"):
            st.session_state["signals"] = cs.mix_strategies(
                list(strategies.values()), mixing_logic
            )
            if st.session_state["signals"] is None:
                center_col3.error(
                    "The signals of the strategies cannot be mixed, please write a correct logic statement."
                )
            else:
                # for m in st.session_state.strategies.values():
                #    st.dataframe(m)
                # st.dataframe(st.session_state["signals"])
                center_col3.success(
                    "The signals of the strategies mixed successfully"
                )
                st.session_state.last_strategy = "Mixed Strategy"


if __name__ == "__main__":
    main()
