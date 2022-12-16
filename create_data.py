import streamlit as st
import datetime as dt
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta import add_all_ta_features
from ta.utils import dropna

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

if "ohlcv" not in st.session_state:
    st.session_state["ohlcv"] = None

def get_financial_data(tickers:str, start:str, end:str, interval:str, auto_adjust:bool) -> pd.DataFrame:
    return yf.download(tickers=tickers, start=start, end=end, interval=interval,
                                auto_adjust=auto_adjust, progress=False)
    

def show_prices(data:pd.DataFrame, ticker:str, show_which_price:str = "All") -> None:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.1, 
                        subplot_titles=(f"Price of '{ticker}'", f"Volume of '{ticker}'"), row_width=[1, 5])
    if show_which_price == "All":
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'],
                                 high=data['High'], low=data['Low'], 
                                 close=data['Close'], name = "OHLC"), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=data.index, y=data[show_which_price], mode='lines', 
                             line=dict(color="#222266"), name=f'{show_which_price} Price'), row=1, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name = "Volume"),
                         row=2, col=1)
    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_layout(autosize=True, width=950, height=950,)
    st.plotly_chart(fig, use_container_width=True)


def signal_smoothing(data:pd.DataFrame, smoothing_method:str= "None", parameters:dict= None):
    if smoothing_method == "None":
        return data
    elif smoothing_method == "Moving Average":
        data["Open"] = data["Open"].rolling(parameters["window"]).mean()
        data["High"] = data["High"].rolling(parameters["window"]).mean()
        data["Low"] = data["Low"].rolling(parameters["window"]).mean()
        data["Close"] = data["Close"].rolling(parameters["window"]).mean()
        data.dropna(inplace=True)
    elif smoothing_method == "Heikin-Ashi":
        data = data.assign(HeikinAshi_Open=np.zeros((data.shape[0])))
        data = data.assign(HeikinAshi_High=np.zeros((data.shape[0])))
        data = data.assign(HeikinAshi_Low=np.zeros((data.shape[0])))
        data = data.assign(HeikinAshi_Close=np.zeros((data.shape[0])))
        data.iloc[0, data.columns.get_loc('HeikinAshi_Open')] = ((data["Open"].iloc[0] + data["Close"].iloc[0]) / 2)
        for i in range(data.shape[0]):
            if i != 0:
                data.iloc[i, data.columns.get_loc('HeikinAshi_Open')] = ((data["HeikinAshi_Open"].iloc[i-1] + data["HeikinAshi_Close"].iloc[i-1]) / 2) 
            data.iloc[i, data.columns.get_loc('HeikinAshi_Close')] = (data["Open"].iloc[i] + data["High"].iloc[i] + data["Low"].iloc[i] + data["Close"].iloc[i]) / 4
            data.iloc[i, data.columns.get_loc('HeikinAshi_High')] = max([data["High"].iloc[i],data["HeikinAshi_Open"].iloc[i],data["HeikinAshi_Close"].iloc[i]])
            data.iloc[i, data.columns.get_loc('HeikinAshi_Low')] = min([data["Low"].iloc[i],data["HeikinAshi_Open"].iloc[i],data["HeikinAshi_Close"].iloc[i]])     
        data = data.iloc[29: , :] 
        data["Open"] = data["HeikinAshi_Open"] 
        data["High"] = data["HeikinAshi_High"]
        data["Low"] = data["HeikinAshi_Low"]
        data["Close"] = data["HeikinAshi_Close"]
    elif smoothing_method == "Trend Normalization":
        data["rowNumber"] = list(range(len(data)))
        data["TN_Open"] = list(range(len(data)))
        data["TN_High"] = list(range(len(data)))
        data["TN_Low"] = list(range(len(data)))
        data["TN_Close"] = list(range(len(data)))
        for i in range(29,len(data)):
            model = LinearRegression()
            model.fit(np.array(data["rowNumber"].iloc[i-29:i+1]).reshape(-1,1), np.array(data["Close"].iloc[i-29:i+1]))
            prediction = model.predict(np.array([data["rowNumber"].iloc[i]]).reshape(-1,1))
            data.iloc[i, data.columns.get_loc("TN_Open")] = data["Open"].iloc[i] - prediction 
            data.iloc[i, data.columns.get_loc("TN_High")] = data["High"].iloc[i] - prediction 
            data.iloc[i, data.columns.get_loc("TN_Low")] = data["Low"].iloc[i] - prediction 
            data.iloc[i, data.columns.get_loc("TN_Close")] = data["Close"].iloc[i] - prediction 
        data["Open"] = data["TN_Open"] 
        data["High"] = data["TN_High"]
        data["Low"] = data["TN_Low"]
        data["Close"] = data["TN_Close"]
        data.drop(index=data.index[:30], axis=0, inplace=True)
    data = data[["Open", "High", "Low", "Close", "Volume"]]
    return data


def create_ohlcv_alike(ohlcv:pd.DataFrame, new_asset:str):
    start = ohlcv.index[0] + dt.timedelta(days=1)
    end = ohlcv.index[-1] + dt.timedelta(days=1)
    interval = "1d" if str(ohlcv.index[1]-ohlcv.index[0]).startswith("1 days") else "1m"
    auto_adjust = False if "Adj Close" in ohlcv.columns else True
    st.write(ohlcv)
    st.write(get_financial_data(tickers = new_asset, start = start, end = end, interval = interval, auto_adjust = auto_adjust))
    return get_financial_data(tickers = new_asset, start = start, end = end, interval = interval, auto_adjust = auto_adjust)




def create_technical_indicators(market:pd.DataFrame) -> pd.DataFrame:
    #market.ta.strategy("All")
    market = add_all_ta_features(market, open="Open", high="High", low="Low", close="Close", volume="Volume") 
    return market

def create_labels(df:pd.DataFrame) -> None:
    df["Label"] = [0] * df.shape[0]
    for i in range(df.shape[0]-10):
        s = set(df["Close"].iloc[i:i+11]) 
        minPrice = sorted(s)[0]
        maxPrice = sorted(s)[-1]
        for j in range(i, i+11):
            if abs(df["Close"].iloc[j] - minPrice) == 0 and (j-i) == 5:
                df.iloc[j, df.columns.get_loc('Label')] = 1
            elif abs(df["Close"].iloc[j] - maxPrice) == 0 and (j-i) == 5:
                df.iloc[j, df.columns.get_loc('Label')] = 2
    df.drop(index=df.index[-6:], axis=0, inplace=True)
    return df

def create_train_test_data(market:pd.DataFrame):
    X_train = []
    X_test = []
    split_point = int(len(market)*0.8)
    selected_feature_count = 30
    select = SelectKBest(score_func=f_classif, k = selected_feature_count)
    fitted = select.fit(market[:split_point].iloc[:,:-1],market[:split_point].iloc[:,-1])
    features = fitted.transform(market[:split_point].iloc[:,:-1])
    X_train.append(features.astype('float32'))
    selected_features_boolean = select.get_support()
    features = list(market.columns[:-1])
    selected_features = []
    for j in range(len(features)):
        if selected_features_boolean[j]:
            selected_features.append(features[j])
    print(f"Selected best {selected_feature_count} features:")
    print(selected_features)
    X_test.append(market[split_point:][selected_features].values.astype('float32'))
    y_train = []
    y_test = []
    y_train.append(market[:split_point]["Label"])
    y_test.append(market[split_point:]["Label"])

    random_forest_model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    sfm = SelectFromModel(random_forest_model, threshold=0.01)
    sfm.fit(X_train[0], y_train[0])
    features=[]
    for feature_list_index in sfm.get_support(indices=True):
        features.append(selected_features[feature_list_index])
    print("Selected features by random forest model: ", features) 
    X_train[0] = sfm.transform(X_train[0])
    X_test[0] = sfm.transform(X_test[0])

    model = LogisticRegression(max_iter=500)
    rfe = RFE(model, n_features_to_select=10)
    fit = rfe.fit(X_train[0], y_train[0])

    new_features=[]
    new_feature_indices=[]
    for i in range(len(fit.support_)):
        if fit.support_[i]:
            new_features.append(features[i])
            new_feature_indices.append(i)
    print("Selected features by recursive feature elimination-logistic regression model: ", new_features)

    for i in range(len(X_train[0][0])-1,-1,-1):
        if i not in new_feature_indices:
            X_train[0] = np.delete(X_train[0], i, axis=1)
            
    for i in range(len(X_test[0][0])-1,-1,-1):
        if i not in new_feature_indices:
            X_test[0] = np.delete(X_test[0], i, axis=1)
    st.write(X_train[0])
    st.write(y_train[0])
    st.write(pd.concat([pd.DataFrame(X_train[0], index=y_train[0].index), y_train[0]], axis=1))
    return X_train, y_train, X_test, y_test


def plot_confusion_matrix(cm, labels, title):
    # cm : confusion matrix list(list)
    # labels : name of the data list(str)
    # title : title for the heatmap
    data = go.Heatmap(z=cm, y=labels, x=labels)
    annotations = []
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            annotations.append(
                {
                    "x": labels[i],
                    "y": labels[j],
                    "font": {"color": "white"},
                    "text": str(value),
                    "xref": "x1",
                    "yref": "y1",
                    "showarrow": False
                }
            )
    layout = {
        "title": title,
        "xaxis": {"title": "Predicted value"},
        "yaxis": {"title": "Real value"},
        "annotations": annotations
    }
    fig = go.Figure(data=data, layout=layout)
    return fig



