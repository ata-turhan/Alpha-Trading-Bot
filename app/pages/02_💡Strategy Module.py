import create_data as cd
import create_strategy as cs
import matplotlib.pyplot as plt
import numpy as np
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
        st.session_state["data"] = None
    if "smoothed_data" not in st.session_state:
        st.session_state["smoothed_data"] = None
    if "strategies" not in st.session_state:
        st.session_state.strategies = {}
    if "predictions" not in st.session_state:
        st.session_state["predictions"] = None
    if "ticker" not in st.session_state:
        st.session_state["ticker"] = ""
    if "mix" not in st.session_state:
        st.session_state.mix = []
    if "added_keys" not in st.session_state:
        st.session_state.added_keys = set()


def clean_predictions():
    st.session_state["predictions"] = None


def main():
    set_session_variables()
    configure_page()
    configure_authors()
    add_bg_from_local("data/background.png", "data/bot.png")

    st.markdown(
        "<h1 style='text-align: center; color: black; font-size: 65px;'> 💡 Strategy Module </h1> <br> <br>",
        unsafe_allow_html=True,
    )
    _, center_col_get, _ = st.columns([1, 2, 1])

    style = "<style>.row-widget.stButton {text-align: center;}</style>"
    st.markdown(style, unsafe_allow_html=True)

    correlated_asset = None

    if st.session_state["data"] is None:
        st.error("Please get the data first.")
    else:
        strategy_fetch_way = center_col_get.selectbox(
            "Which way do you want to get the predictions of a strategy: ",
            ["<Select>", "Create a strategy", "Read from a file"],
        )
        st.markdown("<br> <br>", unsafe_allow_html=True)

        if strategy_fetch_way == "Read from a file":
            uploaded_file = center_col_get.file_uploader(
                "Choose a csv file to upload"
            )
            if uploaded_file is not None:
                try:
                    st.session_state["predictions"] = np.array(
                        pd.read_csv(uploaded_file)
                    )
                except FileNotFoundError as exception:
                    center_col_get.error(
                        "you need to upload a csv or excel file."
                    )
                else:
                    predictions = st.session_state["predictions"]
                    if predictions is not None:
                        st.session_state["strategies"][
                            f"Uploaded Signals-{uploaded_file.name}"
                        ] = st.session_state["predictions"]
                        # st.markdown("<br>", unsafe_allow_html=True)
                        center_col_get.success(
                            "The predictions of strategy fetched successfully"
                        )
        elif strategy_fetch_way == "Create a strategy":
            smooth_method = center_col_get.selectbox(
                "Which way do you want to use the price data with smoothing?",
                [
                    "None",
                    "Moving Average",
                    "Heikin-Ashi",
                    "Trend Normalization",
                ],
            )
            st.session_state["smoothed_data"] = cd.signal_smoothing(
                df=st.session_state["data"],
                smoothing_method=smooth_method,
                parameters={"window": 20},
            )
            strategy_type = center_col_get.selectbox(
                "Which strategy do you want to create: ",
                [
                    "<Select>",
                    "Correlation Trading",
                    "Indicator Trading",
                    "Momentum Trading",
                    "AI Trading",
                    "Candlestick Pattern Trading",
                    "Support-Resistance Trading",
                ],
                on_change=clean_predictions,
            )
            if strategy_type == "Correlation Trading":
                market = center_col_get.selectbox(
                    "Select the correlated market: ",
                    ["<Select>", "Stocks & ETFs", "Forex", "Crypto"],
                )
                if market != "<Select>":
                    assets = list(st.session_state["assets"][market].keys())
                    assets.insert(0, "<Select>")
                    correlated_asset = center_col_get.selectbox(
                        "Select the correlated asset: ", assets
                    )
                    if (
                        correlated_asset is not None
                        and correlated_asset != "<Select>"
                    ):
                        correlated_asset_ohclv = cd.create_ohlcv_alike(
                            data=st.session_state["smoothed_data"],
                            new_asset=st.session_state["assets"][market][
                                correlated_asset
                            ],
                        )
                        try:
                            downward_movement = st.number_input(
                                "How much downward movement do you expect to see from the correlated asset?"
                            )
                        except TypeError:
                            st.write("Please write a number.")
                        if downward_movement != 0:
                            st.markdown("<br>", unsafe_allow_html=True)
                            if st.button(
                                "Create the predictions of the strategy."
                            ):
                                st.session_state[
                                    "predictions"
                                ] = cs.correlation_trading(
                                    data1=correlated_asset_ohclv,
                                    data2=st.session_state["smoothed_data"],
                                    downward_movement=downward_movement,
                                    upward_movement=0.01,
                                )
                                if st.session_state["predictions"] is not None:
                                    st.session_state["predictions"].to_csv(
                                        f"Predictions of the {strategy_type}.csv"
                                    )
                                    st.session_state["strategies"][
                                        f"Correlation Trading-{len(st.session_state['strategies'])}"
                                    ] = st.session_state["predictions"]
                                    st.success(
                                        "Predictions of the strategy created and saved successfully"
                                    )
            elif strategy_type == "Indicator Trading":
                indicator = center_col_get.selectbox(
                    "Select the indicator you want to use: ",
                    ["<Select>", "RSI", "SMA", "EMA", "Bollinger Bands"],
                )
                if indicator != "<Select>":
                    if indicator == "RSI":
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            oversold = st.number_input(
                                "Please enter the oversold value", value=30
                            )
                        with col2:
                            overbought = st.number_input(
                                "Please enter the overbought value", value=70
                            )
                        strategy_created = st.button(
                            "Create the predictions of the strategy."
                        )
                        if strategy_created:
                            st.session_state["predictions"] = cs.rsi_trading(
                                data=st.session_state["smoothed_data"],
                                oversold=oversold,
                                overbought=overbought,
                            )
                    if indicator == "SMA":
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            short_smo = st.number_input(
                                "Please enter the short moving average value",
                                value=50,
                            )
                        with col2:
                            long_smo = st.number_input(
                                "Please enter the long moving average value",
                                value=200,
                            )
                        strategy_created = st.button(
                            "Create the predictions of the strategy."
                        )
                        if strategy_created:
                            st.session_state["predictions"] = cs.sma_trading(
                                data=st.session_state["smoothed_data"],
                                short_mo=short_smo,
                                long_mo=long_smo,
                            )
                    if indicator == "EMA":
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            short_emo = st.number_input(
                                "Please enter the short moving average value",
                                value=50,
                            )
                        with col2:
                            long_emo = st.number_input(
                                "Please enter the long moving average value",
                                value=200,
                            )
                        strategy_created = st.button(
                            "Create the predictions of the strategy."
                        )
                        if strategy_created:
                            st.session_state["predictions"] = cs.ema_trading(
                                data=st.session_state["smoothed_data"],
                                short_mo=short_emo,
                                long_mo=long_emo,
                            )
                    if indicator == "Bollinger Bands":
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            window = st.number_input(
                                "Please enter the window value", value=20
                            )
                        with col2:
                            window_dev = st.number_input(
                                "Please enter the window deviation value",
                                value=2,
                            )
                        strategy_created = st.button(
                            "Create the predictions of the strategy."
                        )
                        st.markdown("<br>", unsafe_allow_html=True)
                        if strategy_created:
                            st.session_state["predictions"] = cs.bb_trading(
                                data=st.session_state["smoothed_data"],
                                window=window,
                                window_dev=window_dev,
                            )
                    if (
                        st.session_state["predictions"] is not None
                        and strategy_created
                    ):
                        st.session_state["predictions"].to_csv(
                            f"Predictions of the {strategy_type}.csv"
                        )
                        st.session_state["strategies"][
                            f"Indicator Trading-{len(st.session_state['strategies'])}"
                        ] = st.session_state["predictions"]
                        st.success(
                            "Predictions of the strategy created and saved successfully"
                        )
                        cs.draw_technical_indicators(
                            data=st.session_state["smoothed_data"],
                            indicator_name=indicator,
                        )
            elif strategy_type == "Momentum Trading":
                indicator = center_col_get.selectbox(
                    "Select the momentum strategy you want to use: ",
                    [
                        "<Select>",
                        "Momentum Day Trading",
                        "Momentum Percentage Trading",
                    ],
                )
                if indicator != "<Select>":
                    if indicator == "Momentum Day Trading":
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col1:
                            up_day = st.number_input(
                                "Please enter the up day number", value=3
                            )
                        with col2:
                            down_day = st.number_input(
                                "Please enter the down day number", value=3
                            )
                        with col3:
                            reverse = st.checkbox(
                                "Reverse the logic of the strategy"
                            )
                        strategy_created = st.button(
                            "Create the predictions of the strategy."
                        )
                        if strategy_created:
                            st.session_state[
                                "predictions"
                            ] = cs.momentum_day_trading(
                                data=st.session_state["smoothed_data"],
                                up_day=up_day,
                                down_day=down_day,
                                reverse=reverse,
                            )
                    if indicator == "Momentum Percentage Trading":
                        col1, col2, col3, col4, col5 = st.columns(
                            [1, 1, 1, 1, 1]
                        )
                        with col1:
                            up_day = st.number_input(
                                "Please enter the up day number", value=3
                            )
                        with col2:
                            up_percentage = st.number_input(
                                "Please enter the up percentage number",
                                value=3,
                            )
                        with col3:
                            down_day = st.number_input(
                                "Please enter the down day number", value=3
                            )
                        with col4:
                            down_percentage = st.number_input(
                                "Please enter the down percentage number",
                                value=3,
                            )
                        with col5:
                            reverse = st.checkbox(
                                "Reverse the logic of the strategy"
                            )
                        st.markdown("<br>", unsafe_allow_html=True)
                        strategy_created = st.button(
                            "Create the predictions of the strategy."
                        )
                        if strategy_created:
                            st.session_state[
                                "predictions"
                            ] = cs.momentum_percentage_trading(
                                data=st.session_state["smoothed_data"],
                                up_percentage=up_percentage,
                                up_day=up_day,
                                down_percentage=down_percentage,
                                down_day=down_day,
                                reverse=reverse,
                            )
                    if (
                        st.session_state["predictions"] is not None
                        and strategy_created
                    ):
                        st.session_state["predictions"].to_csv(
                            f"Predictions of the {strategy_type}.csv"
                        )
                        # st.write(type(st.session_state["strategies"]))
                        st.session_state["strategies"][
                            f"Momentum Trading-{len(st.session_state['strategies'])}"
                        ] = st.session_state["predictions"]
                        st.success(
                            "Predictions of the strategy created and saved successfully"
                        )
            elif strategy_type == "AI Trading":
                data_types = st.multiselect(
                    "Select the data types you want to include for ai modeling: ",
                    [
                        "<Select>",
                        "Fundamental Data",
                        "Technical Data",
                        "Sentiment Data",
                    ],
                    help="Fundamental Data: CPI, DXY, Fed Rate.\nTechnical Data: Technical Indicators\nSentiment Data: \
                    Sentiment of tweets of last 24 hours.",
                )
                if "Sentiment Data" in data_types:
                    transformer_type = center_col_get.selectbox(
                        "Select the tranformer model you want to use: ",
                        ["<Select>", "Vader"],
                    )
                if "Technical Data" in data_types:
                    if (
                        "Label"
                        not in st.session_state["smoothed_data"].columns
                        and "volume_obv"
                        not in st.session_state["smoothed_data"].columns
                    ):
                        with st.spinner("Technical indicators are created..."):
                            st.session_state[
                                "smoothed_data"
                            ] = cd.create_technical_indicators(
                                st.session_state["smoothed_data"]
                            )
                        with st.spinner("True labels are created..."):
                            st.session_state[
                                "smoothed_data"
                            ] = cd.create_labels(
                                st.session_state["smoothed_data"]
                            )
                        # with st.spinner('Train and test data are created...'):
                        # X_train, y_train, X_test, y_test = create_train_test_data(st.session_state["data"])
                    st.success("Technical data is ready!")
                    ai_method = center_col_get.selectbox(
                        "Select the artifical intelligence method you want to use: ",
                        ["<Select>", "Machine Learning", "Deep Learning"],
                    )
                    if ai_method == "Deep Learning":
                        dl_method = center_col_get.selectbox(
                            "Select the deep learning model you want to use: ",
                            ["<Select>", "AutoKeras", "Prophet"],
                        )
                        if dl_method == "AutoKeras":
                            possible_models = st.number_input(
                                "Select the number of the possible models to try",
                                value=5,
                                step=5,
                            )
                            if st.button(
                                "Create the predictions of the strategy."
                            ):
                                market = st.session_state["smoothed_data"]
                                train_data = market.iloc[
                                    : len(market) * 4 // 5, :
                                ]
                                test_data = market.iloc[
                                    len(market) * 4 // 5 :, :
                                ]
                                st.session_state[
                                    "predictions"
                                ] = cs.dl_trading(
                                    train_data=train_data,
                                    test_data=test_data,
                                    possible_models=possible_models,
                                )
                    elif ai_method == "Machine Learning":
                        models = cs.get_ml_models(
                            st.session_state["smoothed_data"].iloc[:100, :]
                        )
                        ai_models = st.multiselect(
                            "Select the machine learning models you want to use: ",
                            models.keys(),
                        )
                        tune_number = st.number_input(
                            "Select the number of the iterations to tune the model",
                            value=5,
                            step=5,
                        )
                        if len(ai_models) != 0:
                            selected_models = [
                                models[key] for key in ai_models
                            ]
                            # st.write(type(selected_models))
                            st.markdown("<br>", unsafe_allow_html=True)
                            if st.button(
                                "Create the predictions of the strategy."
                            ):
                                # train_data = pd.concat([pd.DataFrame(X_train[0], index=y_train[0].index), y_train[0]], axis=1)
                                # test_data = pd.DataFrame(X_test[0], index=y_test[0].index)
                                market = st.session_state["smoothed_data"]
                                train_data = market.iloc[
                                    : len(market) * 4 // 5, :
                                ]
                                test_data = market.iloc[
                                    len(market) * 4 // 5 :, :
                                ]
                                st.session_state[
                                    "predictions"
                                ] = cs.ml_trading(
                                    train_data=train_data,
                                    test_data=test_data,
                                    selected_models=selected_models,
                                    tune_number=tune_number,
                                )
                                if st.session_state["predictions"] is not None:
                                    st.session_state["predictions"].to_csv(
                                        f"Predictions of the {strategy_type}.csv"
                                    )
                                    st.session_state["strategies"][
                                        f"AI Trading-{len(st.session_state['strategies'])}"
                                    ] = st.session_state["predictions"]
                                    st.success(
                                        "Predictions of the strategy created and saved successfully"
                                    )
                                    target_names = ["Hold", "Buy", "Sell"]
                                    cm = confusion_matrix(
                                        test_data["Label"],
                                        st.session_state["predictions"][
                                            "Predictions"
                                        ],
                                    )
                                    plt.grid(False)
                                    disp = ConfusionMatrixDisplay(
                                        confusion_matrix=cm,
                                        display_labels=target_names,
                                    )
                                    disp.plot(cmap=plt.cm.Blues)
                                    disp.ax_.grid(False)
                                    fig = disp.ax_.get_figure()
                                    fig.set_size_inches(5, 5)
                                    col1, col2 = st.columns([1, 1])
                                    with col1:
                                        st.subheader(
                                            "Confusion matrix of the AI Model"
                                        )
                                        st.pyplot(fig)
                                    with col2:
                                        st.subheader(
                                            "Classification report of the AI Model"
                                        )
                                        report = classification_report(
                                            test_data["Label"],
                                            st.session_state["predictions"][
                                                "Predictions"
                                            ],
                                            target_names=target_names,
                                            output_dict=True,
                                        )
                                        df_report = pd.DataFrame(
                                            report
                                        ).transpose()
                                        st.dataframe(df_report)
                                        train_period = pd.DataFrame(
                                            index=train_data.index,
                                            data={
                                                "Predictions": np.zeros(
                                                    (len(train_data),)
                                                )
                                            },
                                        )
                                        st.session_state[
                                            "predictions"
                                        ] = pd.concat(
                                            [
                                                train_period,
                                                st.session_state[
                                                    "predictions"
                                                ],
                                            ]
                                        )
            elif strategy_type == "Candlestick Pattern Trading":
                col1, col2 = st.columns([1, 1])
                with col1:
                    buy_pattern = center_col_get.selectbox(
                        "Select the pattern you want to use for a buy signal:",
                        [
                            "<Select>",
                            "Doji",
                            "Gravestone Doji",
                            "Dragonfly Doji",
                            "Longleg Doji",
                            "Hammer Hanging Man",
                            "Inverse Hammer",
                            "Spinning Top",
                            "Dark Cloud Cover",
                            "Piercing Pattern",
                            "Bullish Marubozu",
                            "Bullish Engulfing",
                            "Bullish Harami",
                        ],
                    )
                with col2:
                    sell_pattern = center_col_get.selectbox(
                        "Select the pattern you want to use for a sell signal:",
                        [
                            "<Select>",
                            "Doji",
                            "Gravestone Doji",
                            "Dragonfly Doji",
                            "Longleg Doji",
                            "Hammer Hanging Man",
                            "Inverse Hammer",
                            "Spinning Top",
                            "Dark Cloud Cover",
                            "Piercing Pattern",
                            "Bearish Marubozu",
                            "Bearish Engulfing",
                            "Bearish Harami",
                        ],
                    )
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Create the predictions of the strategy."):
                    if buy_pattern == "<Select>" or sell_pattern == "<Select>":
                        st.warning(
                            "Please select the patterns for buy and sell signals."
                        )
                    else:
                        st.session_state[
                            "predictions"
                        ] = cs.candlestick_trading(
                            data=st.session_state["smoothed_data"],
                            buy_pattern=buy_pattern,
                            sell_pattern=sell_pattern,
                        )
                        if st.session_state["predictions"] is not None:
                            st.session_state["predictions"].to_csv(
                                f"Predictions of the {strategy_type}.csv"
                            )
                            st.session_state["strategies"][
                                f"Candlestick Pattern Trading-{len(st.session_state['strategies'])}"
                            ] = st.session_state["predictions"]
                            st.success(
                                "Predictions of the strategy created and saved successfully"
                            )
            elif strategy_type == "Support-Resistance Trading":
                col1, col2 = st.columns([1, 1])
                with col1:
                    rolling_wave_length = st.number_input(
                        "Please enter the rolling wave length", value=20
                    )
                with col2:
                    num_clusters = st.number_input(
                        "Please enter the cluster numbers", value=4
                    )
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Create the predictions of the strategy."):
                    st.session_state[
                        "predictions"
                    ] = cs.support_resistance_trading(
                        data=st.session_state["smoothed_data"],
                        rolling_wave_length=rolling_wave_length,
                        num_clusters=num_clusters,
                    )
                    if st.session_state["predictions"] is not None:
                        st.session_state["predictions"].to_csv(
                            f"Predictions of the {strategy_type}.csv"
                        )
                        st.session_state["strategies"][
                            f"Support-Resistance-Trading-{len(st.session_state['strategies'])}"
                        ] = st.session_state["predictions"]
                        st.success(
                            "Predictions of the strategy created and saved successfully"
                        )
            if (
                st.session_state["predictions"] is not None
                and strategy_type != "<Select>"
            ):
                st.write(st.session_state["predictions"])
                predictions = st.session_state["predictions"]
                cs.show_predictions_on_chart(
                    data=st.session_state["smoothed_data"],
                    predictions=predictions,
                    ticker=st.session_state["ticker"],
                )
    if len(st.session_state["strategies"]) != 0:
        strategies = st.session_state["strategies"]
        st.markdown("<br><br>", unsafe_allow_html=True)
        _, center_col_strategies_created, _ = st.columns([1, 2, 1])
        center_col_strategies_created.subheader(
            "These strategies have been created:"
        )
        st.markdown("<br>", unsafe_allow_html=True)

        for key, val in strategies.items():
            _, left_col, _, right_col, _ = st.columns([1, 2, 1, 2, 1])
            with left_col:
                st.write(key)
            with right_col:
                if (
                    st.checkbox("Use this strategy in mix.", key=key)
                    and key not in st.session_state.added_keys
                ):
                    st.session_state.mix.append(strategies[key])
                    st.session_state.added_keys.add(key)

        st.markdown("<br><br>", unsafe_allow_html=True)
        _, center_col3, _ = st.columns([1, 2, 1])
        mixing_logic = center_col3.text_input(
            "Write your logic to mix the strategies with and-or:",
            help="For example: 's1 and s2 and s3'",
        )
        if st.button("Mix the strategies"):
            st.session_state["predictions"] = cs.mix_strategies(
                st.session_state.mix, mixing_logic
            )
            center_col2.success(
                "Predictions of the strategies mixed successfully"
            )


if __name__ == "__main__":
    main()
