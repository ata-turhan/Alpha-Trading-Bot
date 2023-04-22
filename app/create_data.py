import datetime as dt
import json
import random

pass

import numpy as np
import pandas as pd
import pandas_datareader as pdr
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    f_classif,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from ta import add_all_ta_features

fred_codes = {
    "FED 2Y Interest Rate": "DGS2",
    "FED 10Y Interest Rate": "DGS10",
    "30-Year Fixed Rate Mortgage Average in the United States": "MORTGAGE30US",
    "Unemployment Rate": "UNRATE",
    "Real Gross Domestic Product": "GDPC1",
    "Gross Domestic Product": "GDP",
    "10-Year Breakeven Inflation Rate": "T10YIE",
    "Median Sales Price of Houses Sold for the United States": "MSPUS",
    "Personal Saving Rate": "PSAVERT",
    "Deposits, All Commercial Banks": "DPSACBW027SBOG",
    "S&P 500": "SP500",
    "Federal Debt: Total Public Debt as Percent of Gross Domestic Product": "GFDEGDQ188S",
    "Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, Oklahoma": "DCOILWTICO",
    "Consumer Loans: Credit Cards and Other Revolving Plans, All Commercial Banks": "CCLACBW027SBOG",
    "Consumer Price Index for All Urban Consumers: All Items Less Food and Energy in U.S. City Average": "CPILFESL",
}


def get_financial_data(
    tickers: str, start: str, end: str, interval: str, auto_adjust: bool
) -> pd.DataFrame:
    return yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
    )


def show_prices(
    data: pd.DataFrame, ticker: str, show_which_price: str
) -> None:
    colors = ["red", "green", "blue", "orange", "black", "magenta", "cyan"]
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.1,
        subplot_titles=(f"Chart", f"Volume"),
        row_width=[1, 5],
    )
    if "Candlestick" in show_which_price:
        show_which_price.remove("Candlestick")
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="OHLC",
            ),
            row=1,
            col=1,
        )
    for column in show_which_price:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[column],
                mode="lines",
                line=dict(color=random.choice(colors)),
                name=f"{column}",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Bar(x=data.index, y=data["Volume"], name="Volume"), row=2, col=1
    )
    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_layout(
        autosize=True,
        width=950,
        height=950,
    )
    st.plotly_chart(fig, use_container_width=True)


def signal_smoothing(
    df: pd.DataFrame, smoothing_method: str = "None", parameters: dict = None
):
    data = df.copy(deep=True)
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
        data.iloc[0, data.columns.get_loc("HeikinAshi_Open")] = (
            data["Open"].iloc[0] + data["Close"].iloc[0]
        ) / 2
        for i in range(data.shape[0]):
            if i != 0:
                data.iloc[i, data.columns.get_loc("HeikinAshi_Open")] = (
                    data["HeikinAshi_Open"].iloc[i - 1]
                    + data["HeikinAshi_Close"].iloc[i - 1]
                ) / 2
            data.iloc[i, data.columns.get_loc("HeikinAshi_Close")] = (
                data["Open"].iloc[i]
                + data["High"].iloc[i]
                + data["Low"].iloc[i]
                + data["Close"].iloc[i]
            ) / 4
            data.iloc[i, data.columns.get_loc("HeikinAshi_High")] = max(
                [
                    data["High"].iloc[i],
                    data["HeikinAshi_Open"].iloc[i],
                    data["HeikinAshi_Close"].iloc[i],
                ]
            )
            data.iloc[i, data.columns.get_loc("HeikinAshi_Low")] = min(
                [
                    data["Low"].iloc[i],
                    data["HeikinAshi_Open"].iloc[i],
                    data["HeikinAshi_Close"].iloc[i],
                ]
            )
        data.drop(index=data.index[:30], axis=0, inplace=True)
        data["Open"] = data["HeikinAshi_Open"]
        data["High"] = data["HeikinAshi_High"]
        data["Low"] = data["HeikinAshi_Low"]
        data["Close"] = data["HeikinAshi_Close"]
        data.drop(
            [
                "HeikinAshi_Open",
                "HeikinAshi_High",
                "HeikinAshi_Low",
                "HeikinAshi_Close",
            ],
            axis=1,
            inplace=True,
        )
    elif smoothing_method == "Trend Normalization":
        data["rowNumber"] = list(range(len(data)))
        data["TN_Open"] = list(range(len(data)))
        data["TN_High"] = list(range(len(data)))
        data["TN_Low"] = list(range(len(data)))
        data["TN_Close"] = list(range(len(data)))
        for i in range(29, len(data)):
            model = LinearRegression()
            model.fit(
                np.array(data["rowNumber"].iloc[i - 29 : i + 1]).reshape(
                    -1, 1
                ),
                np.array(data["Close"].iloc[i - 29 : i + 1]),
            )
            prediction = model.predict(
                np.array([data["rowNumber"].iloc[i]]).reshape(-1, 1)
            )
            data.iloc[i, data.columns.get_loc("TN_Open")] = (
                data["Open"].iloc[i] - prediction
            )
            data.iloc[i, data.columns.get_loc("TN_High")] = (
                data["High"].iloc[i] - prediction
            )
            data.iloc[i, data.columns.get_loc("TN_Low")] = (
                data["Low"].iloc[i] - prediction
            )
            data.iloc[i, data.columns.get_loc("TN_Close")] = (
                data["Close"].iloc[i] - prediction
            )
        data["Open"] = data["TN_Open"]
        data["High"] = data["TN_High"]
        data["Low"] = data["TN_Low"]
        data["Close"] = data["TN_Close"]
        data.drop(index=data.index[:30], axis=0, inplace=True)
        data.drop(
            [
                "rowNumber",
                "TN_Open",
                "TN_High",
                "TN_Low",
                "TN_Close",
            ],
            axis=1,
            inplace=True,
        )
    return data


def create_ohlcv_alike(data: pd.DataFrame, new_asset: str):
    start = data.index[0] - dt.timedelta(days=1)
    end = data.index[-1] + dt.timedelta(days=1)
    interval = (
        "1d"
        if str(data.index[1] - data.index[0]).startswith("1 days")
        else "1m"
    )
    auto_adjust = "Adj Close" not in data.columns
    return get_financial_data(
        tickers=new_asset,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
    )


def create_technical_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        with st.spinner("Fetching indicator data"):
            data = add_all_ta_features(
                ohlcv,
                open="Open",
                high="High",
                low="Low",
                close="Close",
                volume="Volume",
                colprefix="ti_",
            )
            return data


def create_labels(ohlcv: pd.DataFrame) -> None:
    df = ohlcv.copy()
    df["Label"] = [0] * df.shape[0]
    for i in range(df.shape[0] - 10):
        s = set(df["Close"].iloc[i : i + 11])
        minPrice, maxPrice = sorted(s)[0], sorted(s)[-1]
        for j in range(i, i + 11):
            if df["Close"].iloc[j] == minPrice and (j - i) == 5:
                df.iloc[j, df.columns.get_loc("Label")] = 1
            elif df["Close"].iloc[j] == maxPrice and (j - i) == 5:
                df.iloc[j, df.columns.get_loc("Label")] = 2
    df.drop(index=df.index[-6:], axis=0, inplace=True)
    return df


def create_train_test_data(market: pd.DataFrame):
    split_point = int(len(market) * 0.8)
    selected_feature_count = 30
    select = SelectKBest(score_func=f_classif, k=selected_feature_count)
    fitted = select.fit(
        market[:split_point].iloc[:, :-1], market[:split_point].iloc[:, -1]
    )
    features = fitted.transform(market[:split_point].iloc[:, :-1])
    X_train = [features.astype("float32")]
    selected_features_boolean = select.get_support()
    features = list(market.columns[:-1])
    selected_features = [
        features[j]
        for j in range(len(features))
        if selected_features_boolean[j]
    ]
    print(f"Selected best {selected_feature_count} features:")
    print(selected_features)
    X_test = [market[split_point:][selected_features].values.astype("float32")]
    y_train = [market[:split_point]["Label"]]
    y_test = [market[split_point:]["Label"]]
    random_forest_model = RandomForestClassifier(
        n_estimators=500, random_state=42, n_jobs=-1
    )
    sfm = SelectFromModel(random_forest_model, threshold=0.01)
    sfm.fit(X_train[0], y_train[0])
    features = [
        selected_features[feature_list_index]
        for feature_list_index in sfm.get_support(indices=True)
    ]
    print("Selected features by random forest model: ", features)
    X_train[0] = sfm.transform(X_train[0])
    X_test[0] = sfm.transform(X_test[0])

    model = LogisticRegression(max_iter=500)
    rfe = RFE(model, n_features_to_select=10)
    fit = rfe.fit(X_train[0], y_train[0])

    new_features = []
    new_feature_indices = []
    for i in range(len(fit.support_)):
        if fit.support_[i]:
            new_features.append(features[i])
            new_feature_indices.append(i)
    print(
        "Selected features by recursive feature elimination-logistic regression model: ",
        new_features,
    )

    for i in range(len(X_train[0][0]) - 1, -1, -1):
        if i not in new_feature_indices:
            X_train[0] = np.delete(X_train[0], i, axis=1)

    for i in range(len(X_test[0][0]) - 1, -1, -1):
        if i not in new_feature_indices:
            X_test[0] = np.delete(X_test[0], i, axis=1)
    st.write(X_train[0])
    st.write(y_train[0])
    st.write(
        pd.concat(
            [pd.DataFrame(X_train[0], index=y_train[0].index), y_train[0]],
            axis=1,
        )
    )
    return X_train, y_train, X_test, y_test


def plot_confusion_matrix(cm, labels, title):
    # cm : confusion matrix list(list)
    # labels : name of the data list(str)
    # title : title for the heatmap
    data = go.Heatmap(z=cm, y=labels, x=labels)
    annotations = []
    for i, row in enumerate(cm):
        annotations.extend(
            {
                "x": labels[i],
                "y": labels[j],
                "font": {"color": "white"},
                "text": str(value),
                "xref": "x1",
                "yref": "y1",
                "showarrow": False,
            }
            for j, value in enumerate(row)
        )
    layout = {
        "title": title,
        "xaxis": {"title": "Predicted value"},
        "yaxis": {"title": "Real value"},
        "annotations": annotations,
    }
    return go.Figure(data=data, layout=layout)


def fetch_fred_data(start_date: str, end_date: str) -> pd.DataFrame:
    fred_data = pdr.fred.FredReader(
        symbols=fred_codes.values(), start=start_date, end=end_date
    ).read()
    fred_data.fillna(method="ffill", inplace=True)
    fred_data["10-2 Year Yield Difference"] = (
        fred_data["DGS10"] - fred_data["DGS2"]
    )
    fred_data.rename(
        columns={v: k for k, v in fred_codes.items()},
        inplace=True,
    )
    return fred_data


def fetch_bls_data(data_series_id, start_year, end_year):
    try:
        bls_api_key = "0a2144a551614652b6bc8de8455a2e0c"
        bls_data_api_url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
        headers = {"Content-type": "application/json"}
        data = json.dumps(
            {
                "registrationkey": bls_api_key,
                "seriesid": [data_series_id],
                "startyear": start_year,
                "endyear": end_year,
            }
        )
        response = requests.post(bls_data_api_url, data=data, headers=headers)
        if response.status_code != 200:
            raise IOError(f"Error retrieving data: {response.text}")
        #  Parse JSON
        json_data = json.loads(response.text)
        if json_data["status"] != "REQUEST_SUCCEEDED":
            raise IOError(f"Error retrieving data: {response.text}")
        results_df = pd.DataFrame()
        for series in json_data["Results"]["series"]:
            for item in series["data"]:
                year = int(item["year"])
                period = item["period"]
                period = item["period"]
                year_mon = f"{year}-{int(period.replace('M',''))}"
                date = dt.datetime.strptime(year_mon, "%Y-%m")
                value = float(item["value"])
                row_df = pd.DataFrame({"date": [date], "CPI": [value]})
                results_df = pd.concat([results_df, row_df], ignore_index=True)
        #  Sort ascending
        results_df = results_df.sort_values(by=["date"], ascending=True)
        results_df.set_index(["date"], inplace=True)
        return results_df
    except Exception as ex:
        print(f"Failed to fetch data: {ex}")


def fetch_cpi_data(start_date, end_date):
    data_series_id = "CUUR0000SA0"
    return fetch_bls_data(
        data_series_id, str(start_date)[:4], str(end_date)[:4]
    )


def fetch_fundamental_data(
    data: pd.DataFrame, start_date, end_date
) -> pd.DataFrame:
    fed_data_exists = False
    cpi_data_exists = True
    fundamentals = [
        "FED 2Y Interest Rate",
        "FED 10Y Interest Rate",
        "Yield Difference",
    ]
    for fundamental in fundamentals[:3]:
        if fundamental in data.columns:
            fed_data_exists = True

    data = data.tz_localize(None)
    data.index.names = ["Date"]

    if not fed_data_exists:
        fed_data = fetch_fred_data(
            start_date - pd.DateOffset(months=6), end_date
        )
        data["merg_col"] = data.index.strftime("%Y-%m-%d")
        fed_data["merg_col"] = fed_data.index.strftime("%Y-%m-%d")
        data = (
            data.reset_index()
            .merge(fed_data, on="merg_col", how="left")
            .set_index("Date")
            .drop(columns="merg_col")
        )

    if not cpi_data_exists:
        cpi_data = fetch_cpi_data(
            start_date - pd.DateOffset(months=1), end_date
        )
        data.index = data.index - pd.DateOffset(months=1)
        data["merg_col"] = data.index.strftime("%Y-%m")
        cpi_data["merg_col"] = cpi_data.index.strftime("%Y-%m")
        data = (
            data.reset_index()
            .merge(cpi_data, on="merg_col", how="left")
            .set_index("Date")
            .drop(columns="merg_col")
        )
        data.index = data.index + pd.DateOffset(months=1)
    return data
