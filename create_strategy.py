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


def correlation_trading(ohlcv1:pd.DataFrame, ohlcv2:pd.DataFrame, downward_movement:float=0.01, upward_movement:float=0.01):
    index1 = ohlcv1[ohlcv1.pct_change()["Close"] < -downward_movement].index
    index2 = ohlcv2[ohlcv2.pct_change()["Close"] > -downward_movement].index
    indices = list(set(index1).intersection(set(index2)))
    for idx in range(len(indices)):
        indices[idx] = indices[idx] + dt.timedelta(days=1)
    predictions = pd.DataFrame(index=ohlcv2.index, data={"Predictions":np.zeros((len(ohlcv2),))})
    predictions.loc[indices, "Predictions"] = 1
    return predictions


def rsi_trading(ohlcv:pd.DataFrame, oversold:int=30, overbought:int=70):
    predictions = pd.DataFrame(index=ohlcv.index, data={"Predictions":np.zeros((len(ohlcv),))})
    ohlcv["RSI"] = ta.momentum.RSIIndicator(close = ohlcv["Close"], window  = 14, fillna = False).rsi()
    for i in range(len(ohlcv)-1):
        if ohlcv.loc[ohlcv.index[i], "RSI"] <= oversold:
            predictions.loc[ohlcv.index[i+1], "Predictions"] = 1
        elif ohlcv.loc[ohlcv.index[i], "RSI"] >= overbought:
            predictions.loc[ohlcv.index[i+1], "Predictions"] = 2
    return predictions


def sma_trading(ohlcv:pd.DataFrame, short_mo:int=50, long_mo:int=200):
    predictions = pd.DataFrame(index=ohlcv.index, data={"Predictions":np.zeros((len(ohlcv),))})
    ohlcv[f"SMA-{short_mo}"] = ohlcv['Close'].ewm(span=short_mo).mean()
    ohlcv[f"SMA-{long_mo}"] = ohlcv['Close'].ewm(span=long_mo).mean()
    short_mo_above = False
    for i in range(len(ohlcv)-1):
        if ohlcv.loc[ohlcv.index[i], f"SMA-{short_mo}"] >= ohlcv.loc[ohlcv.index[i], f"SMA-{long_mo}"] and not short_mo_above:
            predictions.loc[ohlcv.index[i+1], "Predictions"] = 1
            short_mo_above = True
        elif ohlcv.loc[ohlcv.index[i], f"SMA-{short_mo}"] <= ohlcv.loc[ohlcv.index[i], f"SMA-{long_mo}"] and short_mo_above:
            predictions.loc[ohlcv.index[i+1], "Predictions"] = 2
            short_mo_above = False
    return predictions


def ema_trading(ohlcv:pd.DataFrame, short_mo:int=50, long_mo:int=200):
    predictions = pd.DataFrame(index=ohlcv.index, data={"Predictions":np.zeros((len(ohlcv),))})
    ohlcv[f"EMA-{short_mo}"] = ohlcv['Close'].ewm(span=short_mo).mean()
    ohlcv[f"EMA-{long_mo}"] = ohlcv['Close'].ewm(span=long_mo).mean()
    short_mo_above = False
    for i in range(len(ohlcv)-1):
        if ohlcv.loc[ohlcv.index[i], f"EMA-{short_mo}"] >= ohlcv.loc[ohlcv.index[i], f"EMA-{long_mo}"] and not short_mo_above:
            predictions.loc[ohlcv.index[i+1], "Predictions"] = 1
            short_mo_above = True
        elif ohlcv.loc[ohlcv.index[i], f"EMA-{short_mo}"] <= ohlcv.loc[ohlcv.index[i], f"EMA-{long_mo}"] and short_mo_above:
            predictions.loc[ohlcv.index[i+1], "Predictions"] = 2
            short_mo_above = False
    return predictions


def bb_trading(ohlcv:pd.DataFrame, window:int=20, window_dev:int=2):
    predictions = pd.DataFrame(index=ohlcv.index, data={"Predictions":np.zeros((len(ohlcv),))})
    indicator_bb = BollingerBands(close=ohlcv["Close"], window=window, window_dev=window_dev)
    ohlcv["bb_bbh"] = indicator_bb.bollinger_hband()
    ohlcv["bb_bbl"] = indicator_bb.bollinger_lband()
    for i in range(len(ohlcv)-1):
        if ohlcv.loc[ohlcv.index[i], "Close"] <= ohlcv.loc[ohlcv.index[i], "bb_bbl"]:
            predictions.loc[ohlcv.index[i+1], "Predictions"] = 1
        elif ohlcv.loc[ohlcv.index[i], "Close"] >= ohlcv.loc[ohlcv.index[i], "bb_bbh"]:
            predictions.loc[ohlcv.index[i+1], "Predictions"] = 2
    return predictions


def momentum_day_trading(ohlcv:pd.DataFrame, up_day:int=3, down_day:int=3, reverse:bool=False):
    predictions = pd.DataFrame(index=ohlcv.index, data={"Predictions":np.zeros((len(ohlcv),))})
    ohlcv["change"] = ohlcv["Close"].pct_change()
    for i in range(1, len(ohlcv)-up_day):
        open_position = True
        for j in range(up_day):
            if ohlcv.at[ohlcv.index[i+j], "change"] <= 0:
                open_position = False
        if open_position:
            predictions.at[ohlcv.index[i+1], "Predictions"] = 1 if not reverse else 2
        open_position = True
        for j in range(down_day):
            if ohlcv.at[ohlcv.index[i+j], "change"] >= 0:
                open_position = False
        if open_position:
            predictions.at[ohlcv.index[i+1], "Predictions"] = 2 if not reverse else 1
    return predictions


def momentum_percentage_trading(ohlcv:pd.DataFrame, up_percentage:int=5, up_day:int=3, down_percentage:int=5, 
                         down_day:int=3, reverse:bool=False):
    predictions = pd.DataFrame(index=ohlcv.index, data={"Predictions":np.zeros((len(ohlcv),))})
    for i in range(1, len(ohlcv)-up_day-1):
        if (ohlcv.at[ohlcv.index[i+up_day], "Close"] - ohlcv.at[ohlcv.index[i], "Close"]) \
                / ohlcv.at[ohlcv.index[i], "Close"] * 100 >= up_percentage:
            predictions.at[ohlcv.index[i+up_day+1], "Predictions"] = 1 if not reverse else 2
        elif (ohlcv.at[ohlcv.index[i+down_day], "Close"] - ohlcv.at[ohlcv.index[i], "Close"]) \
                / ohlcv.at[ohlcv.index[i], "Close"] * 100 <= -down_percentage:
            predictions.at[ohlcv.index[i+down_day+1], "Predictions"] = 2 if not reverse else 1
    return predictions


def ai_trading(ai_model:str, train_data:pd.DataFrame, test_data:pd.DataFrame):
    with st.spinner('Data preprocessing...'):
        s = classification.setup(data = train_data, 
            target = 'Label', 
            experiment_name = 'ai_trading',
            fold = 5,
            use_gpu = False,
            normalize = True,
            pca = False,
            remove_outliers = True,
            remove_multicollinearity = True,
            feature_selection = False,
            fix_imbalance = True,
            silent=True,
            )
    with st.spinner('Create the model...'):
        model = classification.create_model(ai_model)
    with st.spinner('Tune the model...'):
        tuned_model = classification.tune_model(model, optimize = "F1", n_iter = 5, choose_better = True)
    # default model
    print(model)
    print("\n\n")
    # tuned model
    print(tuned_model)
    predictions = classification.predict_model(tuned_model, data = test_data)
    return predictions["Label"]

