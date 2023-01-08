import streamlit as st
import datetime as dt
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pycaret import classification
import ta
from ta.volatility import BollingerBands
from Pattern import *
from sklearn.cluster import AgglomerativeClustering
import re


def correlation_trading(
    ohlcv1: pd.DataFrame,
    ohlcv2: pd.DataFrame,
    downward_movement: float = 0.01,
    upward_movement: float = 0.01,
):
    index1 = ohlcv1[ohlcv1.pct_change()["Close"] < -downward_movement].index
    index2 = ohlcv2[ohlcv2.pct_change()["Close"] > -downward_movement].index
    indices = list(set(index1).intersection(set(index2)))
    for idx in range(len(indices)):
        indices[idx] = indices[idx] + dt.timedelta(days=1)
    predictions = pd.DataFrame(
        index=ohlcv2.index, data={"Predictions": np.zeros((len(ohlcv2),))}
    )
    predictions.loc[indices, "Predictions"] = 1
    return predictions


def rsi_trading(ohlcv: pd.DataFrame, oversold: int = 30, overbought: int = 70):
    predictions = pd.DataFrame(
        index=ohlcv.index, data={"Predictions": np.zeros((len(ohlcv),))}
    )
    ohlcv["RSI"] = ta.momentum.RSIIndicator(
        close=ohlcv["Close"], window=14, fillna=False
    ).rsi()
    for i in range(len(ohlcv) - 1):
        if ohlcv.at[ohlcv.index[i], "RSI"] <= oversold:
            predictions.at[ohlcv.index[i + 1], "Predictions"] = 1
        elif ohlcv.at[ohlcv.index[i], "RSI"] >= overbought:
            predictions.at[ohlcv.index[i + 1], "Predictions"] = 2
    return predictions


def sma_trading(ohlcv: pd.DataFrame, short_mo: int = 50, long_mo: int = 200):
    predictions = pd.DataFrame(
        index=ohlcv.index, data={"Predictions": np.zeros((len(ohlcv),))}
    )
    ohlcv[f"SMA-{short_mo}"] = ohlcv["Close"].rolling(short_mo).mean()
    ohlcv[f"SMA-{long_mo}"] = ohlcv["Close"].rolling(long_mo).mean()
    short_mo_above = False
    for i in range(len(ohlcv) - 1):
        if (
            ohlcv.at[ohlcv.index[i], f"SMA-{short_mo}"]
            >= ohlcv.at[ohlcv.index[i], f"SMA-{long_mo}"]
            and not short_mo_above
        ):
            predictions.at[ohlcv.index[i + 1], "Predictions"] = 1
            short_mo_above = True
        elif (
            ohlcv.at[ohlcv.index[i], f"SMA-{short_mo}"]
            <= ohlcv.at[ohlcv.index[i], f"SMA-{long_mo}"]
            and short_mo_above
        ):
            predictions.at[ohlcv.index[i + 1], "Predictions"] = 2
            short_mo_above = False
    return predictions


def ema_trading(ohlcv: pd.DataFrame, short_mo: int = 50, long_mo: int = 200):
    predictions = pd.DataFrame(
        index=ohlcv.index, data={"Predictions": np.zeros((len(ohlcv),))}
    )
    ohlcv[f"EMA-{short_mo}"] = ohlcv["Close"].ewm(span=short_mo).mean()
    ohlcv[f"EMA-{long_mo}"] = ohlcv["Close"].ewm(span=long_mo).mean()
    short_mo_above = False
    for i in range(len(ohlcv) - 1):
        if (
            ohlcv.at[ohlcv.index[i], f"EMA-{short_mo}"]
            >= ohlcv.at[ohlcv.index[i], f"EMA-{long_mo}"]
            and not short_mo_above
        ):
            predictions.at[ohlcv.index[i + 1], "Predictions"] = 1
            short_mo_above = True
        elif (
            ohlcv.at[ohlcv.index[i], f"EMA-{short_mo}"]
            <= ohlcv.at[ohlcv.index[i], f"EMA-{long_mo}"]
            and short_mo_above
        ):
            predictions.at[ohlcv.index[i + 1], "Predictions"] = 2
            short_mo_above = False
    return predictions


def bb_trading(ohlcv: pd.DataFrame, window: int = 20, window_dev: int = 2):
    predictions = pd.DataFrame(
        index=ohlcv.index, data={"Predictions": np.zeros((len(ohlcv),))}
    )
    indicator_bb = BollingerBands(
        close=ohlcv["Close"], window=window, window_dev=window_dev
    )
    ohlcv["bb_bbh"] = indicator_bb.bollinger_hband()
    ohlcv["bb_bbl"] = indicator_bb.bollinger_lband()
    for i in range(len(ohlcv) - 1):
        if ohlcv.at[ohlcv.index[i], "Close"] <= ohlcv.at[ohlcv.index[i], "bb_bbl"]:
            predictions.at[ohlcv.index[i + 1], "Predictions"] = 1
        elif ohlcv.at[ohlcv.index[i], "Close"] >= ohlcv.at[ohlcv.index[i], "bb_bbh"]:
            predictions.at[ohlcv.index[i + 1], "Predictions"] = 2
    return predictions


def momentum_day_trading(
    ohlcv: pd.DataFrame, up_day: int = 3, down_day: int = 3, reverse: bool = False
):
    predictions = pd.DataFrame(
        index=ohlcv.index, data={"Predictions": np.zeros((len(ohlcv),))}
    )
    ohlcv["change"] = ohlcv["Close"].pct_change()
    for i in range(1, len(ohlcv) - up_day):
        open_position = True
        for j in range(up_day):
            if ohlcv.at[ohlcv.index[i + j], "change"] <= 0:
                open_position = False
        if open_position:
            predictions.at[ohlcv.index[i + up_day], "Predictions"] = (
                1 if not reverse else 2
            )
        open_position = True
        for j in range(down_day):
            if ohlcv.at[ohlcv.index[i + j], "change"] >= 0:
                open_position = False
        if open_position:
            predictions.at[ohlcv.index[i + down_day], "Predictions"] = (
                2 if not reverse else 1
            )
    return predictions


def momentum_percentage_trading(
    ohlcv: pd.DataFrame,
    up_percentage: int = 5,
    up_day: int = 3,
    down_percentage: int = 5,
    down_day: int = 3,
    reverse: bool = False,
):
    predictions = pd.DataFrame(
        index=ohlcv.index, data={"Predictions": np.zeros((len(ohlcv),))}
    )
    for i in range(1, len(ohlcv) - up_day - 1):
        if (
            ohlcv.at[ohlcv.index[i + up_day], "Close"]
            - ohlcv.at[ohlcv.index[i], "Close"]
        ) / ohlcv.at[ohlcv.index[i], "Close"] * 100 >= up_percentage:
            predictions.at[ohlcv.index[i + up_day + 1], "Predictions"] = (
                1 if not reverse else 2
            )
        elif (
            ohlcv.at[ohlcv.index[i + down_day], "Close"]
            - ohlcv.at[ohlcv.index[i], "Close"]
        ) / ohlcv.at[ohlcv.index[i], "Close"] * 100 <= -down_percentage:
            predictions.at[ohlcv.index[i + down_day + 1], "Predictions"] = (
                2 if not reverse else 1
            )
    return predictions


def get_ml_models(train_data: pd.DataFrame):
    s = classification.setup(
        data=train_data,
        target="Label",
        silent=True,
    )
    df = classification.models()
    df.query('Name != "Dummy Classifier"', inplace=True)
    ids = df.index
    names = df.Name
    models = {name: id_val for id_val, name in zip(ids, names)}
    return models


def ai_trading(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    selected_models: list,
    tune_number: int,
):
    with st.spinner("Data preprocessing..."):
        s = classification.setup(
            data=train_data,
            target="Label",
            experiment_name="ai_trading",
            fold=5,
            use_gpu=False,
            normalize=True,
            pca=False,
            remove_outliers=True,
            remove_multicollinearity=True,
            feature_selection=False,
            fix_imbalance=True,
            silent=True,
        )
    with st.spinner("Create the models..."):
        model = classification.compare_models(include=selected_models)
    with st.spinner("Tune the best model..."):
        tuned_model = classification.tune_model(
            model, optimize="F1", n_iter=tune_number, choose_better=True
        )
    with st.spinner("Finalizing the best model..."):
        final_model = classification.finalize_model(tuned_model)
    # default model
    model_predictions = classification.predict_model(tuned_model, data=test_data)
    predictions = pd.DataFrame(
        index=test_data.index, data={"Predictions": model_predictions["Label"]}
    )
    return predictions


def candlestick_trading(ohlcv: pd.DataFrame, buy_pattern: str, sell_pattern: str):
    candlestick_func_dict = {
        "Doji": doji,
        "Gravestone Doji": gravestone_doji,
        "Dragonfly Doji": dragonfly_doji,
        "Longleg Doji": longleg_doji,
        "Hammer Hanging Man": Hammer_Hanging_Man,
        "Inverse Hammer": Inv_Hammer,
        "Spinning Top": Spinning_Top,
        "Dark Cloud Cover": DarkCloudCover,
        "Piercing Pattern": PiercingPattern,
        "Bullish Marubozu": Marubozu,
        "Bearish Marubozu": Marubozu,
        "Bullish Engulfing": Engulf,
        "Bearish Engulfing": Engulf,
        "Bullish Harami": Harami,
        "Bearish Harami": Harami,
    }
    candlestick_column_dict = {
        "Doji": "Doji",
        "Gravestone Doji": "Gravestone",
        "Dragonfly Doji": "Dragonfly",
        "Longleg Doji": "LongLeg",
        "Hammer Hanging Man": "Hammer",
        "Inverse Hammer": "Inv_Hammer",
        "Spinning Top": "Spinning",
        "Dark Cloud Cover": "DarkCloud",
        "Piercing Pattern": "Piercing",
        "Bullish Marubozu": "Bull_Marubozu",
        "Bearish Marubozu": "Bear_Marubouzu",
        "Bullish Engulfing": "BullEngulf",
        "Bearish Engulfing": "BearEngulf",
        "Bullish Harami": "BullHarami",
        "Bearish Harami": "BearHarami",
    }
    candlestick_func_dict[buy_pattern](ohlcv)
    candlestick_func_dict[sell_pattern](ohlcv)
    predictions = pd.DataFrame(
        index=ohlcv.index, data={"Predictions": np.zeros((len(ohlcv),))}
    )
    for i in range(len(ohlcv) - 1):
        if ohlcv.at[ohlcv.index[i], candlestick_column_dict[buy_pattern]] == True:
            predictions.at[ohlcv.index[i + 1], "Predictions"] = 1
        elif ohlcv.at[ohlcv.index[i], candlestick_column_dict[sell_pattern]] == True:
            predictions.at[ohlcv.index[i + 1], "Predictions"] = 2
    return predictions


def calculate_support_resistance(df, rolling_wave_length, num_clusters, area):
    date = df.index
    # Reset index for merging
    df.reset_index(inplace=True)
    # Create min and max waves
    max_waves_temp = df.High.rolling(rolling_wave_length).max().rename("waves")
    min_waves_temp = df.Low.rolling(rolling_wave_length).min().rename("waves")
    max_waves = pd.concat(
        [max_waves_temp, pd.Series(np.zeros(len(max_waves_temp)) + 1)], axis=1
    )
    min_waves = pd.concat(
        [min_waves_temp, pd.Series(np.zeros(len(min_waves_temp)) + -1)], axis=1
    )
    #  Remove dups
    max_waves.drop_duplicates("waves", inplace=True)
    min_waves.drop_duplicates("waves", inplace=True)
    #  Merge max and min waves
    waves = pd.concat([max_waves, min_waves]).sort_index()
    waves = waves[waves[0] != waves[0].shift()].dropna()
    # Find Support/Resistance with clustering using the rolling stats
    # Create [x,y] array where y is always 1
    x = np.concatenate(
        (waves.waves.values.reshape(-1, 1), (np.zeros(len(waves)) + 1).reshape(-1, 1)),
        axis=1,
    )
    # Initialize Agglomerative Clustering
    cluster = AgglomerativeClustering(
        n_clusters=num_clusters, affinity="euclidean", linkage="ward"
    )
    cluster.fit_predict(x)
    waves["clusters"] = cluster.labels_
    # Get index of the max wave for each cluster
    if area == "resistance":
        waves2 = waves.loc[waves.groupby("clusters")["waves"].idxmax()]
    if area == "support":
        waves2 = waves.loc[waves.groupby("clusters")["waves"].idxmin()]
    df.index = date
    df.drop("Date", axis=1, inplace=True)
    waves2.waves.drop_duplicates(keep="first", inplace=True)
    return waves2.reset_index().waves


def support_resistance_trading(
    ohlcv: pd.DataFrame, rolling_wave_length: int = 20, num_clusters: int = 4
):
    predictions = pd.DataFrame(
        index=ohlcv.index, data={"Predictions": np.zeros((len(ohlcv),))}
    )
    supports = calculate_support_resistance(
        ohlcv, rolling_wave_length, num_clusters, "support"
    )
    resistances = calculate_support_resistance(
        ohlcv, rolling_wave_length, num_clusters, "resistance"
    )
    for i in range(rolling_wave_length * 2 + num_clusters, len(ohlcv) - 1):
        for support in supports:
            if (
                ohlcv.at[ohlcv.index[i - 1], "Close"] >= support
                and ohlcv.at[ohlcv.index[i], "Close"] < support
            ):
                predictions.at[ohlcv.index[i + 1], "Predictions"] = 2
        if predictions.at[ohlcv.index[i + 1], "Predictions"] == 0:
            for resistance in resistances:
                if (
                    ohlcv.at[ohlcv.index[i - 1], "Close"] <= resistance
                    and ohlcv.at[ohlcv.index[i], "Close"] > resistance
                ):
                    predictions.at[ohlcv.index[i + 1], "Predictions"] = 1
    return predictions


def show_predictions_on_chart(ohlcv: pd.DataFrame, predictions: np.array, ticker: str):
    fig = go.Figure()
    buy_labels = predictions == 1
    sell_labels = predictions == 2
    fig.add_trace(
        go.Scatter(
            x=ohlcv.index,
            y=ohlcv["Close"],
            mode="lines",
            line=dict(color="#222266"),
            name="Close Price",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ohlcv[buy_labels].index,
            y=ohlcv[buy_labels]["Close"],
            mode="markers",
            marker=dict(size=6, color="#2cc05c"),
            name="Buy",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ohlcv[sell_labels].index,
            y=ohlcv[sell_labels]["Close"],
            mode="markers",
            marker=dict(size=6, color="#f62728"),
            name="Sell",
        )
    )
    fig.update_layout(
        title=f"<span style='font-size: 30px;'><b>Close Price with Predictions of {ticker}</b></span>",
        title_x=0.5,
    )
    st.plotly_chart(fig, use_container_width=True)


def mix_strategies(mix: set, mixing_logic: str):
    mix_prediction = np.zeros((len(mix[0]),))
    mixing_logic = re.sub(r"s[0-9]*", "{}", mixing_logic)
    for m in mix:
        m["is_buy"] = m["Predictions"] == 1
        m["is_sell"] = m["Predictions"] == 2
    for i in range(len(mix[0])):
        buy_evaluations = []
        sell_evaluations = []
        for m in mix:
            buy_evaluations.append(m["is_buy"].iat[i])
            sell_evaluations.append(m["is_sell"].iat[i])
        buy_evaluation = eval(mixing_logic.format(*buy_evaluations))
        sell_evaluation = eval(mixing_logic.format(*sell_evaluations))
        # st.write(buy_evaluation)
        if buy_evaluation:
            mix_prediction[i] = 1
        if sell_evaluation:
            mix_prediction[i] = 2
    predictions = pd.DataFrame(index=mix[0].index, data={"Predictions": mix_prediction})
    return predictions


def draw_technical_indicators(ohlcv: pd.DataFrame, indicator_name: str):
    if indicator_name == "Bollinger Bands":
        indicator_bb = BollingerBands(close=ohlcv["Close"], window=20, window_dev=2)
        ohlcv["bb_bbh"] = indicator_bb.bollinger_hband()
        ohlcv["bb_bbl"] = indicator_bb.bollinger_lband()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ohlcv.index,
                y=ohlcv["Close"],
                mode="lines",
                line=dict(color="#222266"),
                name="Close Price",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ohlcv.index,
                y=ohlcv["bb_bbh"],
                mode="lines",
                line=dict(color="#2cc05c"),
                name="Bollinger Higher Band",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ohlcv.index,
                y=ohlcv["bb_bbl"],
                mode="lines",
                line=dict(color="#f62728"),
                name="Bollinger Lower Band",
            )
        )
        fig.update_layout(
            title=f"<span style='font-size: 30px;'><b>Close Price with Bollinger Bands</b></span>",
            title_x=0.5,
        )
        st.plotly_chart(fig, use_container_width=True)
    elif indicator_name == "RSI":
        ohlcv["RSI"] = ta.momentum.RSIIndicator(
            close=ohlcv["Close"], window=14, fillna=False
        ).rsi()
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.1,
            subplot_titles=(
                f"<span style='font-size: 30px;'><b>Close Price</b></span>",
                f"<span style='font-size: 30px;'><b>RSI Value</b></span>",
            ),
            row_width=[1, 1],
        )
        fig.add_trace(
            go.Scatter(
                x=ohlcv.index,
                y=ohlcv["Close"],
                mode="lines",
                line=dict(color="#222266"),
                name="Close Price",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=ohlcv.index,
                y=ohlcv["RSI"],
                mode="lines",
                line=dict(color="#880099"),
                name="RSI Value",
            ),
            row=2,
            col=1,
        )
        fig.update_layout(
            title=f"<span style='font-size: 30px;'><b>Close Price with RSI</b></span>",
            title_x=0.5,
        )
        fig.update_layout(
            autosize=True,
            width=950,
            height=950,
        )
        st.plotly_chart(fig, use_container_width=True)
    elif indicator_name == "SMA":
        ohlcv["SMA-short"] = ohlcv["Close"].rolling(50).mean()
        ohlcv["SMA-long"] = ohlcv["Close"].rolling(200).mean()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ohlcv.index,
                y=ohlcv["Close"],
                mode="lines",
                line=dict(color="#222266"),
                name="Close Price",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ohlcv.index,
                y=ohlcv["SMA-short"],
                mode="lines",
                line=dict(color="#2cc05c"),
                name="Short SMA",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ohlcv.index,
                y=ohlcv["SMA-long"],
                mode="lines",
                line=dict(color="#f62728"),
                name="Long SMA",
            )
        )
        fig.update_layout(
            title=f"<span style='font-size: 30px;'><b>Close Price with SMAs</b></span>",
            title_x=0.5,
        )
        st.plotly_chart(fig, use_container_width=True)
    elif indicator_name == "EMA":
        ohlcv["EMA-short"] = ohlcv["Close"].ewm(span=50).mean()
        ohlcv["EMA-long"] = ohlcv["Close"].ewm(span=200).mean()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ohlcv.index,
                y=ohlcv["Close"],
                mode="lines",
                line=dict(color="#222266"),
                name="Close Price",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ohlcv.index,
                y=ohlcv["EMA-short"],
                mode="lines",
                line=dict(color="#2cc05c"),
                name="Short SMA",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ohlcv.index,
                y=ohlcv["EMA-long"],
                mode="lines",
                line=dict(color="#f62728"),
                name="Long SMA",
            )
        )
        fig.update_layout(
            title=f"<span style='font-size: 30px;'><b>Close Price with EMAs</b></span>",
            title_x=0.5,
        )
        st.plotly_chart(fig, use_container_width=True)
