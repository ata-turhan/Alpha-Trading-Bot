import streamlit as st
import datetime as dt
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pycaret import classification


def correlation_trading(ohlcv1:pd.DataFrame, ohlcv2:pd.DataFrame, downward_movement:float=0.01, upward_movement:float=0.01):
    index1 = ohlcv1[ohlcv1.pct_change()["Close"] < -downward_movement].index
    index2 = ohlcv2[ohlcv2.pct_change()["Close"] > -downward_movement].index
    indices = list(set(index1).intersection(set(index2)))
    for idx in range(len(indices)):
        indices[idx] = indices[idx] + dt.timedelta(days=1)
    predictions = pd.DataFrame(index=ohlcv2.index, data={"Predictions":np.zeros((len(ohlcv2),))})
    predictions.loc[indices, "Predictions"] = 1
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

