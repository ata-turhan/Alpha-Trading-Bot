import datetime as dt
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import ta
from Pattern import (
    DarkCloudCover,
    Engulf,
    Hammer_Hanging_Man,
    Harami,
    Inv_Hammer,
    Marubozu,
    PiercingPattern,
    Spinning_Top,
    doji,
    dragonfly_doji,
    gravestone_doji,
    longleg_doji,
)
from plotly.subplots import make_subplots

# from pycaret import classification
from sklearn.cluster import AgglomerativeClustering
from ta.volatility import BollingerBands

# import autokeras as ak
# import tensorflow as tf
# import keras
# from keras.callbacks import (Callback,ModelCheckpoint,EarlyStopping,CSVLogger,ReduceLROnPlateau,)


candlestick_bullish_patterns = [
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
]


candlestick_bearish_patterns = [
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
]


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
    signals = pd.DataFrame(
        index=ohlcv2.index, data={"Signals": np.zeros((len(ohlcv2),))}
    )
    signals.loc[indices, "Signals"] = 1
    return signals


def rsi_trading(ohlcv: pd.DataFrame, oversold: int = 30, overbought: int = 70):
    signals = pd.DataFrame(
        index=ohlcv.index, data={"Signals": np.zeros((len(ohlcv),))}
    )
    ohlcv["RSI"] = ta.momentum.RSIIndicator(
        close=ohlcv["Close"], window=14, fillna=False
    ).rsi()
    for i in range(len(ohlcv) - 1):
        if ohlcv.at[ohlcv.index[i], "RSI"] <= oversold:
            signals.at[ohlcv.index[i + 1], "Signals"] = 1
        elif ohlcv.at[ohlcv.index[i], "RSI"] >= overbought:
            signals.at[ohlcv.index[i + 1], "Signals"] = 2
    return signals


def sma_trading(ohlcv: pd.DataFrame, short_smo: int = 50, long_smo: int = 200):
    signals = pd.DataFrame(
        index=ohlcv.index, data={"Signals": np.zeros((len(ohlcv),))}
    )
    ohlcv[f"SMA-{short_smo}"] = ohlcv["Close"].rolling(short_smo).mean()
    ohlcv[f"SMA-{long_smo}"] = ohlcv["Close"].rolling(long_smo).mean()
    short_smo_above = False
    for i in range(len(ohlcv) - 1):
        if (
            ohlcv.at[ohlcv.index[i], f"SMA-{short_smo}"]
            >= ohlcv.at[ohlcv.index[i], f"SMA-{long_smo}"]
            and not short_smo_above
        ):
            signals.at[ohlcv.index[i + 1], "Signals"] = 1
            short_smo_above = True
        elif (
            ohlcv.at[ohlcv.index[i], f"SMA-{short_smo}"]
            <= ohlcv.at[ohlcv.index[i], f"SMA-{long_smo}"]
            and short_smo_above
        ):
            signals.at[ohlcv.index[i + 1], "Signals"] = 2
            short_smo_above = False
    return signals


def ema_trading(ohlcv: pd.DataFrame, short_emo: int = 50, long_emo: int = 200):
    signals = pd.DataFrame(
        index=ohlcv.index, data={"Signals": np.zeros((len(ohlcv),))}
    )
    ohlcv[f"EMA-{short_emo}"] = ohlcv["Close"].ewm(span=short_emo).mean()
    ohlcv[f"EMA-{long_emo}"] = ohlcv["Close"].ewm(span=long_emo).mean()
    short_emo_above = False
    for i in range(len(ohlcv) - 1):
        if (
            ohlcv.at[ohlcv.index[i], f"EMA-{short_emo}"]
            >= ohlcv.at[ohlcv.index[i], f"EMA-{long_emo}"]
            and not short_emo_above
        ):
            signals.at[ohlcv.index[i + 1], "Signals"] = 1
            short_emo_above = True
        elif (
            ohlcv.at[ohlcv.index[i], f"EMA-{short_emo}"]
            <= ohlcv.at[ohlcv.index[i], f"EMA-{long_emo}"]
            and short_emo_above
        ):
            signals.at[ohlcv.index[i + 1], "Signals"] = 2
            short_emo_above = False
    return signals


def bb_trading(ohlcv: pd.DataFrame, window: int = 20, window_dev: int = 2):
    signals = pd.DataFrame(
        index=ohlcv.index, data={"Signals": np.zeros((len(ohlcv),))}
    )
    indicator_bb = BollingerBands(
        close=ohlcv["Close"], window=window, window_dev=window_dev
    )
    ohlcv["bb_bbh"] = indicator_bb.bollinger_hband()
    ohlcv["bb_bbl"] = indicator_bb.bollinger_lband()
    for i in range(len(ohlcv) - 1):
        if (
            ohlcv.at[ohlcv.index[i], "Close"]
            <= ohlcv.at[ohlcv.index[i], "bb_bbl"]
        ):
            signals.at[ohlcv.index[i + 1], "Signals"] = 1
        elif (
            ohlcv.at[ohlcv.index[i], "Close"]
            >= ohlcv.at[ohlcv.index[i], "bb_bbh"]
        ):
            signals.at[ohlcv.index[i + 1], "Signals"] = 2
    return signals


def momentum_day_trading(
    ohlcv: pd.DataFrame,
    up_day: int = 3,
    down_day: int = 3,
    reverse: bool = False,
):
    signals = pd.DataFrame(
        index=ohlcv.index, data={"Signals": np.zeros((len(ohlcv),))}
    )
    ohlcv["change"] = ohlcv["Close"].pct_change()
    for i in range(1, len(ohlcv) - up_day):
        long_position = all(
            ohlcv.at[ohlcv.index[i + j], "change"] > 0 for j in range(up_day)
        )
        if long_position:
            signals.at[ohlcv.index[i + up_day], "Signals"] = (
                2 if reverse else 1
            )
        short_position = all(
            ohlcv.at[ohlcv.index[i + j], "change"] < 0 for j in range(down_day)
        )
        if short_position:
            signals.at[ohlcv.index[i + down_day], "Signals"] = (
                1 if reverse else 2
            )
    # st.write(signals)
    return signals


def momentum_percentage_trading(
    ohlcv: pd.DataFrame,
    up_percentage: int = 5,
    up_day: int = 3,
    down_percentage: int = 5,
    down_day: int = 3,
    reverse: bool = False,
):
    signals = pd.DataFrame(
        index=ohlcv.index, data={"Signals": np.zeros((len(ohlcv),))}
    )
    for i in range(1, len(ohlcv) - up_day - 1):
        if (
            ohlcv.at[ohlcv.index[i + up_day], "Close"]
            - ohlcv.at[ohlcv.index[i], "Close"]
        ) / ohlcv.at[ohlcv.index[i], "Close"] * 100 >= up_percentage:
            signals.at[ohlcv.index[i + up_day + 1], "Signals"] = (
                2 if reverse else 1
            )
        elif (
            ohlcv.at[ohlcv.index[i + down_day], "Close"]
            - ohlcv.at[ohlcv.index[i], "Close"]
        ) / ohlcv.at[ohlcv.index[i], "Close"] * 100 <= -down_percentage:
            signals.at[ohlcv.index[i + down_day + 1], "Signals"] = (
                1 if reverse else 2
            )
    return signals


"""
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


@st.cache
def ml_trading(
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
    # default model
    model_signals = classification.predict_model(
        tuned_model, data=test_data
    )
    return pd.DataFrame(
        index=test_data.index, data={"Signals": model_signals["Label"]}
    )
"""

"""
def dl_trading(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    possible_models: list,
):
    callback = [
        EarlyStopping(monitor="val_loss", patience=25, mode="min"),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=15,
            verbose=1,
            min_delta=1e-3,
            mode="min",
        ),
    ]
    with st.spinner("Searching best neural network architecture..."):
        search = ak.StructuredDataClassifier(
            max_trials=possible_models,
            project_name="Asset Price Classifier",
            overwrite=True,
        )
        search.fit(
            x=train_data.loc[:, :"Label"],
            y=train_data.loc[:, "Label"],
            epochs=30,
            verbose=2,
            callbacks=[callback],
            validation_split=0.3,
        )
    with st.spinner("Finalizing the best model..."):
        best_nn_model = search.export_model()
    preds = best_nn_model.predict(test_data.loc[:, :"Label"])
    signals = np.argmax(preds, axis=1)
    return pd.DataFrame(
        index=test_data.index, data={"Signals": signals}
    )
"""


def candlestick_pattern_trading(
    ohlcv: pd.DataFrame, buy_pattern: str, sell_pattern: str
):
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
    signals = pd.DataFrame(
        index=ohlcv.index, data={"Signals": np.zeros((len(ohlcv),))}
    )
    for i in range(len(ohlcv) - 1):
        if (
            ohlcv.at[ohlcv.index[i], candlestick_column_dict[buy_pattern]]
            == True
        ):
            signals.at[ohlcv.index[i + 1], "Signals"] = 1
        elif (
            ohlcv.at[ohlcv.index[i], candlestick_column_dict[sell_pattern]]
            == True
        ):
            signals.at[ohlcv.index[i + 1], "Signals"] = 2
    return signals


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
        (
            waves.waves.values.reshape(-1, 1),
            (np.zeros(len(waves)) + 1).reshape(-1, 1),
        ),
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
    signals = pd.DataFrame(
        index=ohlcv.index, data={"Signals": np.zeros((len(ohlcv),))}
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
                signals.at[ohlcv.index[i + 1], "Signals"] = 2
        if signals.at[ohlcv.index[i + 1], "Signals"] == 0:
            for resistance in resistances:
                if (
                    ohlcv.at[ohlcv.index[i - 1], "Close"] <= resistance
                    and ohlcv.at[ohlcv.index[i], "Close"] > resistance
                ):
                    signals.at[ohlcv.index[i + 1], "Signals"] = 1
    return signals


def classify_candle(open, high, low, close):
    if open > close:
        body_color = "filled"
    else:
        body_color = "hollow"
    body_height = abs(close - open)
    wick_height = high - max(open, close)
    shadow_height = body_height - wick_height
    if body_height > 2 * wick_height:
        sentiment = "bullish" if body_color == "hollow" else "bearish"
    elif body_height > wick_height:
        sentiment = "bullish" if body_color == "hollow" else "bearish"
    elif wick_height > body_height:
        sentiment = "bearish" if body_color == "hollow" else "bullish"
    elif wick_height < body_height / 2:
        sentiment = (
            "very bullish" if body_color == "hollow" else "very bearish"
        )
    elif shadow_height < body_height / 2:
        sentiment = "bullish" if body_color == "hollow" else "bearish"
    elif shadow_height > body_height / 2:
        sentiment = "bullish" if body_color == "hollow" else "bearish"
    else:
        sentiment = "neutral"
    return sentiment


def candlestick_sentiment_trading(
    ohlcv: pd.DataFrame,
    consecutive_bullish_num: int,
    consecutive_bearish_num: int,
):
    signals = pd.DataFrame(
        index=ohlcv.index, data={"Signals": np.zeros((len(ohlcv),))}
    )
    for i in range(len(ohlcv) - consecutive_bullish_num):
        buy = True
        for j in range(i, i + consecutive_bullish_num):
            row = ohlcv.iloc[j, :]
            open, high, low, close = (
                row["Open"],
                row["High"],
                row["Low"],
                row["Close"],
            )
            sentiment = classify_candle(open, high, low, close)
            if "bullish" not in sentiment:
                buy = False
                break
        if buy:
            signals.at[ohlcv.index[i + consecutive_bullish_num], "Signals"] = 1

    for i in range(len(ohlcv) - consecutive_bearish_num):
        sell = True
        for j in range(i, i + consecutive_bearish_num):
            row = ohlcv.iloc[j, :]
            open, high, low, close = (
                row["Open"],
                row["High"],
                row["Low"],
                row["Close"],
            )
            sentiment = classify_candle(open, high, low, close)
            if "bearish" not in sentiment:
                sell = False
                break
        if sell:
            signals.at[ohlcv.index[i + consecutive_bearish_num], "Signals"] = 2
    return signals


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def basic_ml_trading(bml_model: str, train: pd.DataFrame, test: pd.DataFrame):
    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)
    X_train, y_train = train.drop(["Label"], axis=1), train["Label"]
    X_test, y_test = test.drop(["Label"], axis=1), test["Label"]
    # st.dataframe(y_train)
    # st.dataframe(y_test)
    if bml_model == "Logistic Regression":
        model = LogisticRegression()
    elif bml_model == "Support Vector Machine":
        model = SVC()
    elif bml_model == "Random Forest":
        model = RandomForestClassifier()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    st.dataframe(pred)
    full_data = train.append(test)
    signals = pd.DataFrame(
        index=full_data.index, data={"Signals": np.zeros((len(full_data),))}
    )
    signals.loc[test.index[0] :, "Signals"] = pred
    return signals


def dl_trading(dl_model_layer: str, train: pd.DataFrame, test: pd.DataFrame):
    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)
    X_train, y_train = train.drop(["Label"], axis=1), train["Label"]
    X_test, y_test = test.drop(["Label"], axis=1), test["Label"]
    # st.dataframe(y_train)
    # st.dataframe(y_test)
    if bml_model == "Logistic Regression":
        model = LogisticRegression()
    elif bml_model == "Support Vector Machine":
        model = SVC()
    elif bml_model == "Random Forest":
        model = RandomForestClassifier()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    st.dataframe(pred)
    full_data = train.append(test)
    signals = pd.DataFrame(
        index=full_data.index, data={"Signals": np.zeros((len(full_data),))}
    )
    signals.loc[test.index[0] :, "Signals"] = pred
    return signals


def show_signals_on_chart(
    ohlcv: pd.DataFrame,
    signals: np.array,
    last_strategy_name: str,
):
    ohlcv = ohlcv.copy()
    ohlcv["buy"] = np.where(signals["Signals"] == 1, 1, 0)
    ohlcv["sell"] = np.where(signals["Signals"] == 2, 1, 0)
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
            x=ohlcv[ohlcv["buy"] == True].index,
            y=ohlcv[ohlcv["buy"] == True]["Close"],
            mode="markers",
            marker=dict(size=6, color="#2cc05c"),
            name="Buy",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ohlcv[ohlcv["sell"] == True].index,
            y=ohlcv[ohlcv["sell"] == True]["Close"],
            mode="markers",
            marker=dict(size=6, color="#f62728"),
            name="Sell",
        )
    )
    fig.update_layout(
        title=f"<span style='font-size: 30px;'><b>Close Price with the Signals of the {last_strategy_name}</b></span>",
        title_x=0.05,
        autosize=True,
        width=950,
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def mix_strategies(mix: set, mixing_logic: str):
    try:
        mix_signal = np.zeros((len(mix[0]),))
        mixing_logic = re.sub(
            r"\bs(\d+)\b", lambda match: f"{{{match.group(1)}}}", mixing_logic
        )
        for m in mix:
            m["is_buy"] = np.where(m["Signals"] == 1, 1, 0)
            m["is_sell"] = np.where(m["Signals"] == 2, 1, 0)
        for i in range(len(mix[0])):
            buy_evaluations = [
                0,
            ]
            sell_evaluations = [
                0,
            ]
            for m in mix:
                buy_evaluations.append(m["is_buy"].iat[i])
                sell_evaluations.append(m["is_sell"].iat[i])
            buy_evaluation = eval(mixing_logic.format(*buy_evaluations))
            sell_evaluation = eval(mixing_logic.format(*sell_evaluations))
            if buy_evaluation:
                mix_signal[i] = 1
            if sell_evaluation:
                mix_signal[i] = 2
        return pd.DataFrame(index=mix[0].index, data={"Signals": mix_signal})
    except:
        return None


def draw_technical_indicators(ohlcv: pd.DataFrame, indicator_name: str):
    if indicator_name == "Bollinger Bands":
        indicator_bb = BollingerBands(
            close=ohlcv["Close"], window=20, window_dev=2
        )
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
            title="<span style='font-size: 30px;'><b>Close Price with Bollinger Bands</b></span>",
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
            title="<span style='font-size: 30px;'><b>Close Price with EMAs</b></span>",
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
            vertical_spacing=0.2,
            subplot_titles=(
                "<span style='font-size: 30px;'><b>RSI Value</b></span>",
                "<span style='font-size: 30px;'><b>Close Price</b></span>",
            ),
            row_width=[1, 1],
        )
        fig.add_trace(
            go.Scatter(
                x=ohlcv.index,
                y=ohlcv["RSI"],
                mode="lines",
                line=dict(color="#880099"),
                name="RSI Value",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=ohlcv.index,
                y=ohlcv["Close"],
                mode="lines",
                line=dict(color="#222266"),
                name="Close Price",
            ),
            row=2,
            col=1,
        )
        fig.update_layout(
            autosize=True,
            width=950,
            height=600,
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
            title="<span style='font-size: 30px;'><b>Close Price with SMAs</b></span>",
            title_x=0.5,
        )
        st.plotly_chart(fig, use_container_width=True)
