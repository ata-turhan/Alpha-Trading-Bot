import streamlit as st
import datetime as dt
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from create_data import *
from create_strategy import *
import base64


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

st.set_page_config(page_title='Trading Bot', 
page_icon='🤖', layout="wide")
st.markdown("<h1 style='text-align: center; color: black;'> 💡 Strategy Module </h1> <br> <br>", unsafe_allow_html=True)

add_bg_from_local('data/background.png')


if "ohlcv" not in st.session_state:
    st.session_state["ohlcv"] = None
if "strategies" not in st.session_state:
    st.session_state["strategies"] = []
if "predictions" not in st.session_state:
    st.session_state["predictions"] = None
if "ticker" not in st.session_state:
    st.session_state["ticker"] = ""
correlated_asset = None


for _ in range(22):
    st.sidebar.text("\n")
st.sidebar.write('Developed by Ata Turhan')
st.sidebar.write('Contact at ataturhan21@gmail.com')

if st.session_state["ohlcv"] is None:
    st.error("Please get the data first.")
else:
    strategy_fetch_way = st.selectbox("Which way do you want to get the predictions of a strategy: ", ["<Select>", "Create a strategy", "Read from a file"])
    st.markdown("<br> <br>", unsafe_allow_html=True)

    if strategy_fetch_way == "Read from a file":
        uploaded_file = st.file_uploader("Choose a csv file to upload")
        if uploaded_file is not None:
            try:
                st.session_state["predictions"] = np.array(pd.read_csv(uploaded_file))
            except:
                st.error("you need to upload a csv or excel file.")
            else:
                predictions = st.session_state["predictions"]
                if predictions is not None:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.success("The predictions of strategy fetched successfully")
    elif strategy_fetch_way == "Create a strategy":
        strategy_type = st.selectbox("Which strategy do you want to create: ", ["<Select>", "Correlation Trading", 
        "Indicator Trading", "Momentum Trading", "AI Trading", "Harmonic Trading", "Candlestick Pattern Trading"])
        if strategy_type == "Correlation Trading":
            market = st.selectbox("Select the correlated market: ", ["<Select>",'Stocks & ETFs', 'Forex', "Crypto"])
            if market != "<Select>":
                assets = list(st.session_state["assets"][market].keys())
                assets.insert(0, "<Select>")
                correlated_asset = st.selectbox("Select the correlated asset: ", assets)
                if correlated_asset is not None and correlated_asset != "<Select>":
                    correlated_asset_ohclv = create_ohlcv_alike(ohlcv = st.session_state["ohlcv"], new_asset = st.session_state["assets"][market][correlated_asset])
                    try:
                        downward_movement = st.number_input("How much downward movement do you expect to see from the correlated asset?")
                    except:
                        st.write("Please write a number.")
                    if downward_movement != 0:
                        if st.button("Create the predictions of the strategy."):
                            st.session_state["predictions"] = correlation_trading(ohlcv1 = correlated_asset_ohclv, ohlcv2 = st.session_state["ohlcv"], downward_movement = downward_movement, upward_movement = 0.01)
                            if st.session_state["predictions"] is not None:
                                st.session_state["predictions"].to_csv(f"Predictions of the {strategy_type}.csv")
                                st.success("Predictions of the strategy created and saved successfully")     
        elif strategy_type == "Indicator Trading":    
            indicator = st.selectbox("Select the indicator you want to use: ", ["<Select>",'RSI', 'SMA', "EMA", "Bollinger Bands"])
            if indicator != "<Select>":
                if indicator == "RSI":
                    col1, col2= st.columns([1,1])
                    with col1:
                        oversold = st.number_input("Please enter the oversold value", value = 30)   
                    with col2:
                        overbought = st.number_input("Please enter the overbought value", value = 70) 
                    if st.button("Create the predictions of the strategy."):    
                        st.session_state["predictions"] = rsi_trading(ohlcv = st.session_state["ohlcv"], oversold = oversold,
                                                                 overbought = overbought)
                if indicator == "SMA":
                    col1, col2= st.columns([1,1])
                    with col1:
                        short_smo = st.number_input("Please enter the short moving average value", value = 50)   
                    with col2:
                        long_smo = st.number_input("Please enter the long moving average value", value = 200) 
                    strategy_created = st.button("Create the predictions of the strategy.")
                    if strategy_created:    
                        st.session_state["predictions"] = sma_trading(ohlcv = st.session_state["ohlcv"], short_mo = short_smo,
                                                                 long_mo = long_smo)
                if indicator == "EMA":
                    col1, col2= st.columns([1,1])
                    with col1:
                        short_emo = st.number_input("Please enter the short moving average value", value = 50)   
                    with col2:
                        long_emo = st.number_input("Please enter the long moving average value", value = 200) 
                    strategy_created = st.button("Create the predictions of the strategy.")
                    if strategy_created:    
                        st.session_state["predictions"] = ema_trading(ohlcv = st.session_state["ohlcv"], short_mo = short_emo,
                                                                 long_mo = long_emo)
                if indicator == "Bollinger Bands":
                    col1, col2= st.columns([1,1])
                    with col1:
                        window = st.number_input("Please enter the window value", value = 20)   
                    with col2:
                        window_dev = st.number_input("Please enter the window deviation value", value = 2) 
                    strategy_created = st.button("Create the predictions of the strategy.")
                    if strategy_created:    
                        st.session_state["predictions"] = bb_trading(ohlcv = st.session_state["ohlcv"], window = window,
                                                                 window_dev = window_dev)
                if st.session_state["predictions"] is not None and strategy_created: 
                    st.session_state["predictions"].to_csv(f"Predictions of the {strategy_type}.csv")
                    st.success("Predictions of the strategy created and saved successfully")        
        elif strategy_type == "AI Trading":
            analysis_type = st.selectbox("Select the analyse type you want to apply ai for: ", 
                                        ["<Select>","Technical Analysis", "Sentiment Analysis"])
            if analysis_type == "Sentiment Analysis":
                transformer_type = st.selectbox("Select the tranformer model you want to use: ", 
                                        ["<Select>","Vader"])
                if transformer_type == "Vader":
                    pass
            if analysis_type == "Technical Analysis":
                    if "Label" not in st.session_state["ohlcv"].columns and "volume_obv" not in st.session_state["ohlcv"].columns:
                        with st.spinner('Technical indicators are created...'):
                            st.session_state["ohlcv"] = create_technical_indicators(st.session_state["ohlcv"])
                        with st.spinner('True labels are created...'):
                            st.session_state["ohlcv"] = create_labels(st.session_state["ohlcv"])
                        #with st.spinner('Train and test data are created...'):
                            #X_train, y_train, X_test, y_test = create_train_test_data(st.session_state["ohlcv"]) 
                    st.success('Technical data is ready!')
                    ai_method = st.selectbox("Select the artifical intelligence method you want to use: ", 
                                            ["<Select>", "Machine Learning", "Deep Learning"])
                    if ai_method == "Machine Learning":
                        ai_model = st.selectbox("Select the machine learning model you want to use: ",     
                                                ["<Select>","Extreme Gradient Boosting", "Light Gradient Boosting Machine",
                                                "CatBoost Classifier", "MLP Classifier", "Logistic Regression", "Ada Boost Classifier", 
                                                 "Random Forest Classifier", "Gradient Boosting Classifier", "Extra Trees Classifier", 
                                                 "Decision Tree Classifier", "Quadratic Discriminant Analysis", "K Neighbors Classifier",
                                                 "Gaussian Process Classifier", "SVM - Radial Kernel"])
                        pycaret_abbreviations = {"Extreme Gradient Boosting":"xgboost", "Light Gradient Boosting Machine":"lightgbm",
                                                "CatBoost Classifier":"catboost", "MLP Classifier":"mlp",
                                                 "Logistic Regression":"lr", "Ada Boost Classifier":"ada", 
                                                 "Random Forest Classifier":"rf", "Gradient Boosting Classifier":"gbc", 
                                                 "Extra Trees Classifier":"et", "Decision Tree Classifier":"dt",
                                                 "Quadratic Discriminant Analysis":"qda", "K Neighbors Classifier":"knn",
                                                 "Gaussian Process Classifier":"gpc", "SVM - Radial Kernel":"rbfsvm"}
                        if ai_model != "<Select>":
                            if st.button("Create the predictions of the strategy."):
                                #train_data = pd.concat([pd.DataFrame(X_train[0], index=y_train[0].index), y_train[0]], axis=1)
                                #test_data = pd.DataFrame(X_test[0], index=y_test[0].index)
                                market = st.session_state["ohlcv"] 
                                train_data = market.iloc[:len(market)*4//5,:]
                                test_data = market.iloc[len(market)*4//5:,:]
                                st.session_state["predictions"] = ai_trading(ai_model = pycaret_abbreviations[ai_model], train_data=train_data, test_data=test_data)                            
                                if st.session_state["predictions"] is not None:
                                    st.session_state["predictions"].to_csv(f"Predictions of the {strategy_type}.csv")
                                    st.success("Predictions of the strategy created and saved successfully") 
                        




