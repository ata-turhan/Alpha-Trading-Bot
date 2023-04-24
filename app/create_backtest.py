import base64
import datetime as dt

pass
import random
import time
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import quantstats as qs
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

returns_names = [
    "Cumulative Return",
    "CAGR﹪",
    "Expected Daily",
    "Expected Monthly",
    "Expected Yearly",
    "Best Day",
    "Worst Day",
    "Best Month",
    "Worst Month",
    "Best Year",
    "Worst Year",
    "Avg. Up Month",
    "Avg. Down Month",
]
ratios_names = [
    "Alpha",
    "Beta",
    "Correlation",
    "Sharpe",
    "Prob. Sharpe Ratio",
    "Smart Sharpe",
    "Sortino",
    "Sortino/√2",
    "Smart Sortino/√2",
    "Treynor Ratio",
    "Omega",
    "R^2",
    "Information Ratio",
    "Payoff Ratio",
    "Profit Factor",
    "Common Sense Ratio",
    "CPC Index",
    "Tail Ratio",
    "Outlier Win Ratio",
    "Outlier Loss Ratio",
    "Ulcer Index",
    "Serenity Index",
]
risks_names = [
    "Max Drawdown",
    "Avg. Drawdown",
    "Avg. Drawdown Days",
    "Longest DD Days",
    "Volatility (ann.)",
    "Skew",
    "Kurtosis",
    "Kelly Criterion",
    "Risk of Ruin",
    "Daily Value-at-Risk",
    "Expected Shortfall (cVaR)",
]
counts_names = [
    "Win Days",
    "Win Month",
    "Win Quarter",
    "Win Year",
]
extremums_names = [
    "Max Consecutive Wins",
    "Max Consecutive Losses",
]


def qs_metrics(
    strategy_returns, benchmark_returns, risk_free_rate: int = 0.03
):
    metrics_df = qs.reports.metrics(
        returns=strategy_returns,
        benchmark=benchmark_returns,
        rf=risk_free_rate,
        display=False,
        mode="full",
        sep=False,
        compounded=True,
    )
    metrics_dict = dict()
    metrics_dict["returns"] = metrics_df[metrics_df.index.isin(returns_names)]
    metrics_dict["ratios"] = metrics_df[metrics_df.index.isin(ratios_names)]
    metrics_dict["risks"] = metrics_df[metrics_df.index.isin(risks_names)]
    metrics_dict["counts"] = metrics_df[metrics_df.index.isin(counts_names)]
    metrics_dict["extremums"] = metrics_df[
        metrics_df.index.isin(extremums_names)
    ]
    return metrics_dict, metrics_df


def qs_plots(strategy_returns, benchmark_returns, figsize: tuple = (7, 7)):
    plots_dict = dict()
    plots_dict["snapshot"] = qs.plots.snapshot(
        strategy_returns,
        title=f"{st.session_state['ticker']} Performance",
        figsize=figsize,
        show=False,
        mode="full",
    )

    plots_dict["heatmap"] = qs.plots.monthly_heatmap(
        strategy_returns,
        figsize=figsize,
        show=False,
    )
    plots_dict["normal_returns"] = qs.plots.returns(
        strategy_returns,
        benchmark_returns,
        figsize=figsize,
        show=False,
    )
    plots_dict["log_returns"] = qs.plots.log_returns(
        strategy_returns,
        benchmark_returns,
        figsize=figsize,
        show=False,
    )
    plots_dict["yearly_returns"] = qs.plots.yearly_returns(
        strategy_returns,
        benchmark_returns,
        figsize=figsize,
        show=False,
    )
    plots_dict["daily_returns"] = qs.plots.daily_returns(
        strategy_returns,
        figsize=figsize,
        show=False,
    )
    plots_dict["histogram"] = qs.plots.histogram(
        strategy_returns,
        figsize=figsize,
        show=False,
    )
    plots_dict["distribution"] = qs.plots.distribution(
        strategy_returns,
        figsize=figsize,
        show=False,
    )
    plots_dict["rolling_beta"] = qs.plots.rolling_beta(
        strategy_returns,
        benchmark_returns,
        figsize=figsize,
        show=False,
    )
    plots_dict["rolling_volatility"] = qs.plots.rolling_volatility(
        strategy_returns,
        benchmark_returns,
        figsize=figsize,
        show=False,
    )
    plots_dict["rolling_sharpe"] = qs.plots.rolling_sharpe(
        strategy_returns,
        figsize=figsize,
        show=False,
    )
    plots_dict["rolling_sortino"] = qs.plots.rolling_sortino(
        strategy_returns,
        figsize=figsize,
        show=False,
    )
    plots_dict["drawdowns_periods"] = qs.plots.drawdowns_periods(
        strategy_returns,
        figsize=figsize,
        show=False,
    )
    plots_dict["drawdown"] = qs.plots.drawdown(
        strategy_returns,
        figsize=figsize,
        show=False,
    )
    return plots_dict


def generate_qs_plots_report(plots_dict, charts_dict, report_name):
    report_html = ""
    html_first = """<html>
                <head>
                    <style>
                    {
                        box-sizing: border-box;
                    }
                    /* Set additional styling options for the columns*/
                    .column {
                    float: left;
                    width: 50%;
                    }
                    .row:after {
                    content: "";
                    display: table;
                    clear: both;
                    }
                    .center {
                    display: block;
                    margin-left: auto;
                    margin-right: auto;
                    width: 80%;
                    }
                    </style>
                </head>
                <body>
                """
    report_name = "Backtest QS Plots - " + report_name
    header = f"<h1 style='text-align: center; color: black; font-size: 65px;'> \
    {report_name} </h1> <br>"
    html_first += header
    report_html += html_first

    row_plots = ""
    for i, plot in enumerate(charts_dict.values()):
        tmpfile = BytesIO()
        plot.write_image(tmpfile, format="png")

        encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")
        html = (
            "<br>"
            + "<img class='center' src='data:image/png;base64,{}'>".format(
                encoded
            )
            + "<br>"
        )
        row_plots += html
    html_middle = f"""
    <div class="row">
    {row_plots}
    </div>
    """
    report_html += html_middle

    col1_plots = ""
    col2_plots = ""
    for i, plot in enumerate(plots_dict.values()):
        tmpfile = BytesIO()
        plot.savefig(tmpfile, format="png")
        encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")
        html = (
            "<br>"
            + "<img class='center' src='data:image/png;base64,{}'>".format(
                encoded
            )
            + "<br>"
        )
        if i % 2 == 0:
            col1_plots += html
        else:
            col2_plots += html
    html_last = f"""
                        <div class="row">
                            <div class="column">
                                {col1_plots}
                            </div>
                            <div class="column">
                                {col2_plots}
                            </div>
                        </div>
                    </body>
                    </html>"""
    report_html += html_last
    report_name += ".html"
    with open(report_name, "w") as f:
        f.write(report_html)


def adjustPrices(ohlcv: pd.DataFrame) -> None:
    adjustedRatio = ohlcv["Adj Close"] / ohlcv["Close"]
    ohlcv["High"] = ohlcv["High"] * adjustedRatio
    ohlcv["Low"] = ohlcv["Low"] * adjustedRatio
    ohlcv["Open"] = ohlcv["Open"] * adjustedRatio
    ohlcv["Close"] = ohlcv["Close"] * adjustedRatio


def second_2_minute_converter(seconds: int) -> str:
    minutes = seconds // 60
    return f"{int(minutes)} minutes and {round(seconds%60, 2)} seconds"


def plot_init(
    ticker: str,
    benchmark_ticker: str,
    risk_free_rate: float,
    start_date: list,
    end_date: list,
    initial_capital: float,
    commission: float,
    alpha: float,
    threshold: float,
    order: int,
    short: bool,
    short_fee: float,
    trailing_take_profit: bool,
    take_profit: float,
    trailing_stop_loss: bool,
    stop_loss: float,
    leverage: int,
) -> None:
    configurations = [
        "Ticker",
        "Benchmark Ticker",
        "Risk-free Rate",
        "Start Date",
        "End Date",
        "Inital Capital",
        "Commission",
        "Alpha",
        "Threshold",
        "Order",
        "Short",
        "Short Fee",
        "Trailing Take Profit",
        "Take Profit",
        "Trailing Stop Loss",
        "Stop Loss",
        "Leverage Ratio",
    ]
    values = [
        f"{ticker}",
        f"{benchmark_ticker}",
        f"%{risk_free_rate*100}",
        start_date,
        end_date,
        f"{initial_capital}$",
        f"{commission}$",
        alpha,
        threshold,
        order,
        "is used" if short else "is not used",
        short_fee,
        "is used" if trailing_take_profit else "is not used",
        f"%{take_profit}",
        "is used" if trailing_stop_loss else "is not used",
        f"%{stop_loss}",
        leverage,
    ]
    initial_conf_df = pd.DataFrame(columns=["Configurations", "Values"])
    for i in range(len(configurations)):
        initial_conf_df = initial_conf_df.append(
            pd.Series(
                {
                    "Configurations": configurations[i],
                    "Values": values[i],
                }
            ),
            ignore_index=True,
        )
    # st.dataframe(initial_conf_df, width=500)
    return initial_conf_df


def plot_charts(
    ticker: str,
    ohlcv: pd.DataFrame,
    transactions: np.array,
    portfolio_value: np.array,
    liquidated: bool,
) -> None:
    if liquidated:
        st.write(
            "\n----------------------------\nThis strategy is liquidated\n-------------------------------"
        )
    charts_dict = dict()
    fig = go.Figure()
    buy_labels = transactions == 1
    sell_labels = transactions == 2
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
        title=f"<span style='font-size: 30px;'><b>Close Price with Transactions of {ticker}</b></span>",
        title_x=0.5,
        title_xanchor="center",
        autosize=True,
        width=950,
        height=400,
    )
    charts_dict["transactions"] = fig

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ohlcv.index,
            y=portfolio_value,
            mode="lines",
            line=dict(color="#222266"),
            name="Portfolio Value",
        )
    )
    fig.update_layout(
        title="<span style='font-size: 30px;'><b>Portfolio Value</b></span>",
        title_x=0.5,
        title_xanchor="center",
        autosize=True,
        width=950,
        height=400,
    )
    charts_dict["portfolio_value"] = fig

    return charts_dict


def financial_evaluation(
    hold_label: int = 0,
    buy_label: int = 1,
    sell_label: int = 2,
    ticker: str = "SPY",
    benchmark_ticker: str = "SPY",
    ohlcv: pd.DataFrame = None,
    predictions: np.array = None,
    risk_free_rate: float = 0.03,
    initial_capital: float = 1000,
    commission: float = 1,
    alpha: float = 0.05,
    threshold: float = 0,
    order: int = 1,
    order_type: str = "market",
    short: bool = False,
    short_fee: float = 1,
    trailing_take_profit: bool = False,
    take_profit: float = 10,
    trailing_stop_loss: bool = False,
    stop_loss: float = 10,
    leverage: int = 1,
    miss_rate: int = 10,
    show_initial_configuration: bool = True,
    show_tables: bool = True,
    show_charts: bool = True,
    show_time: bool = False,
    precision_point: int = 3,
) -> dict:
    start = time.time()
    if ohlcv.empty:
        st.write("OHLCV data is empty")
        return
    if predictions.all() is None:
        st.write("Predictions data is empty")
        return
    start_date = ohlcv.index[0] - dt.timedelta(days=5)
    end_date = ohlcv.index[-1] + dt.timedelta(days=5)
    benchmark = yf.download(
        benchmark_ticker,
        start_date,
        end_date,
        progress=False,
        interval="1d",
        auto_adjust=True,
    )
    # st.write(ohlcv)
    # st.write(benchmark_index)
    benchmark = benchmark.loc[ohlcv.index]
    # adjustPrices(ohlcv)
    open_prices = ohlcv["Open"].values
    high_prices = ohlcv["High"].values
    low_prices = ohlcv["Low"].values
    close_prices = ohlcv["Close"].values
    capital = initial_capital
    long_open = False
    short_open = False
    long_price = 0.0
    short_price = 0.0
    stop_loss_price = 0.0
    take_profit_price = 0.0
    total_day_position_open = 0
    total_trade_made = 0
    portfolio_value = np.zeros(len(predictions) + 1)
    portfolio_value[0] = initial_capital
    liquidated = False
    transactions = np.zeros((len(predictions),))
    for i, value in enumerate(predictions):
        change = 0
        if not short:
            if long_open == True and (
                low_prices[i] <= stop_loss_price <= high_prices[i]
                or low_prices[i] <= take_profit_price <= high_prices[i]
            ):
                predictions[i] = sell_label
            if predictions[i] == buy_label and long_open == False:
                if order_type == "market":
                    long_open = True
                    long_price = round(
                        random.uniform(low_prices[i], high_prices[i]), 6
                    )
                elif (
                    order_type == "limit" and random.randint(1, miss_rate) != 1
                ):
                    long_open = True
                    long_price = open_prices[i]
                if long_open == True:
                    stop_loss_price = long_price * (1 - stop_loss / 100)
                    take_profit_price = long_price * (1 + take_profit / 100)
                    capital -= commission
                    transactions[i] = buy_label
            elif predictions[i] == sell_label and long_open == True:
                if order_type == "market":
                    long_open = False
                    change = (
                        (
                            round(
                                random.uniform(low_prices[i], high_prices[i]),
                                6,
                            )
                            - long_price
                        )
                        / long_price
                        * leverage
                    )
                elif (
                    order_type == "limit" and random.randint(1, miss_rate) != 1
                ):
                    long_open = False
                    change = (
                        (open_prices[i] - long_price) / long_price * leverage
                    )
                if long_open == False:
                    capital *= 1 + change
                    capital -= commission
                    transactions[i] = sell_label
                    if capital <= 0:
                        liquidated = True
                        break
                    total_trade_made += 1
            if long_open == True:
                total_day_position_open += 1
            portfolio_value[i + 1] = capital
            if long_open == True and trailing_stop_loss:
                stop_loss_price = close_prices[i] * (1 - stop_loss / 100)
            if long_open == True and trailing_take_profit:
                take_profit_price = close_prices[i] * (1 + take_profit / 100)
        else:
            if (
                predictions[i] != 0
                and long_open == False
                and short_open == False
            ):
                if predictions[i] == buy_label:
                    if order_type == "market":
                        long_open = True
                        long_price = round(
                            random.uniform(low_prices[i], high_prices[i]), 6
                        )
                    elif (
                        order_type == "limit"
                        and random.randint(1, miss_rate) != 1
                    ):
                        long_open = True
                        long_price = open_prices[i]
                    if long_open == True:
                        capital -= commission
                        total_trade_made += 1
                        stop_loss_price = long_price * (1 - stop_loss / 100)
                        take_profit_price = long_price * (
                            1 + take_profit / 100
                        )
                elif predictions[i] == sell_label:
                    if order_type == "market":
                        short_open = True
                        short_price = round(
                            random.uniform(low_prices[i], high_prices[i]), 6
                        )
                    elif (
                        order_type == "limit"
                        and random.randint(1, miss_rate) != 1
                    ):
                        short_open = True
                        short_price = open_prices[i]
                    if short_open == True:
                        capital -= commission + short_fee
                        total_trade_made += 1
                        stop_loss_price = short_price * (1 + stop_loss / 100)
                        take_profit_price = short_price * (
                            1 - take_profit / 100
                        )
            if long_open == True and (
                low_prices[i] <= stop_loss_price <= high_prices[i]
                or low_prices[i] <= take_profit_price <= high_prices[i]
            ):
                predictions[i] = sell_label
            if short_open == True and (
                low_prices[i] <= stop_loss_price <= high_prices[i]
                or low_prices[i] <= take_profit_price <= high_prices[i]
            ):
                predictions[i] = buy_label
            if (
                predictions[i] == sell_label
                and long_open == True
                and short_open == False
            ):
                if order_type == "market":
                    long_open = False
                    short_open = True
                    short_price = round(
                        random.uniform(low_prices[i], high_prices[i]), 6
                    )
                    change = (short_price - long_price) / long_price * leverage
                elif (
                    order_type == "limit" and random.randint(1, miss_rate) != 1
                ):
                    long_open = False
                    short_open = True
                    short_price = open_prices[i]
                    change = (short_price - long_price) / long_price * leverage
                if long_open == False and short_open == True:
                    capital *= 1 + change
                    capital -= commission + short_fee
                    if capital <= 0:
                        liquidated = True
                        break
                    total_trade_made += 1
                    stop_loss_price = short_price * (1 + stop_loss / 100)
                    take_profit_price = short_price * (1 - take_profit / 100)
            elif (
                predictions[i] == buy_label
                and long_open == False
                and short_open == True
            ):
                if order_type == "market":
                    long_open = True
                    short_open = False
                    long_price = round(
                        random.uniform(low_prices[i], high_prices[i]), 6
                    )
                    change = (
                        (short_price - long_price) / short_price * leverage
                    )
                elif (
                    order_type == "limit" and random.randint(1, miss_rate) != 1
                ):
                    long_open = True
                    short_open = False
                    long_price = open_prices[i]
                    change = (
                        (short_price - long_price) / short_price * leverage
                    )
                if long_open == True and short_open == False:
                    capital *= 1 + change
                    capital -= commission
                    if capital <= 0:
                        liquidated = True
                        break
                    total_trade_made += 1
                    stop_loss_price = long_price * (1 - stop_loss / 100)
                    take_profit_price = long_price * (1 + take_profit / 100)
            if long_open == True or short_open == True:
                total_day_position_open += 1
            portfolio_value[i + 1] = capital
            if (
                trailing_stop_loss
                and short_open == False
                and long_open == True
            ):
                stop_loss_price = close_prices[i] * (1 - stop_loss / 100)
            if (
                trailing_stop_loss
                and short_open == True
                and long_open == False
            ):
                stop_loss_price = close_prices[i] * (1 + stop_loss / 100)
            if (
                trailing_take_profit
                and short_open == False
                and long_open == True
            ):
                take_profit_price = close_prices[i] * (1 + take_profit / 100)
            if (
                trailing_take_profit
                and short_open == True
                and long_open == False
            ):
                take_profit_price = close_prices[i] * (1 - take_profit / 100)
    if total_trade_made == 0:
        st.write("No trade executed")
        return
    end = time.time()
    if show_time:
        st.markdown("<br>", unsafe_allow_html=True)
        _, center_col, _ = st.columns([1, 5, 1])
        center_col.subheader(
            f"\nBacktest was completed in {second_2_minute_converter(end-start)}.\n"
        )
    initial_conf_df = plot_init(
        ticker,
        benchmark_ticker,
        risk_free_rate,
        start_date,
        end_date,
        initial_capital,
        commission,
        alpha,
        threshold,
        order,
        short,
        short_fee,
        trailing_take_profit,
        take_profit,
        trailing_stop_loss,
        stop_loss,
        leverage,
    )
    charts_dict_params = [
        "SPY",
        ohlcv[:i],
        transactions[:i],
        portfolio_value[:i],
        liquidated,
    ]
    portfolio = pd.DataFrame(
        portfolio_value[1:], columns=["Value"], index=ohlcv.index
    )
    return portfolio, benchmark, charts_dict_params


def metric_optimization(
    metric: str,
    take_profit_value: float,
    stop_loss_value: float,
    leverage_value: float,
):
    hold_label: int = st.session_state["backtest_configuration"]["hold_label"]
    buy_label: int = st.session_state["backtest_configuration"]["buy_label"]
    sell_label: int = st.session_state["backtest_configuration"]["sell_label"]
    ticker: str = st.session_state["ticker"]
    benchmark_ticker: str = st.session_state["backtest_configuration"][
        "benchmark_ticker"
    ]
    ohlcv: pd.DataFrame = st.session_state["ohlcv"]
    predictions: np.array = np.array(st.session_state["predictions"])
    risk_free_rate: float = st.session_state["backtest_configuration"][
        "risk_free_rate"
    ]
    initial_capital: float = st.session_state["backtest_configuration"][
        "initial_capital"
    ]
    commission: float = st.session_state["backtest_configuration"][
        "commission"
    ]
    alpha: float = st.session_state["backtest_configuration"]["alpha"]
    threshold: float = st.session_state["backtest_configuration"]["threshold"]
    order: int = st.session_state["backtest_configuration"]["order"]
    order_type: str = st.session_state["backtest_configuration"]["order_type"]
    short: bool = st.session_state["backtest_configuration"]["short"]
    short_fee: float = st.session_state["backtest_configuration"]["short_fee"]
    trailing_take_profit: bool = st.session_state["backtest_configuration"][
        "trailing_take_profit"
    ]
    take_profit: float = take_profit_value
    trailing_stop_loss: bool = st.session_state["backtest_configuration"][
        "trailing_stop_loss"
    ]
    stop_loss: float = stop_loss_value
    leverage: int = leverage_value
    miss_rate: int = st.session_state["backtest_configuration"]["miss_rate"]
    show_initial_configuration: bool = False
    show_tables: bool = False
    show_charts: bool = False
    show_time: bool = False
    precision_point = st.session_state["backtest_configuration"][
        "precision_point"
    ]

    metrics = financial_evaluation(
        hold_label,
        buy_label,
        sell_label,
        ticker,
        benchmark_ticker,
        ohlcv,
        predictions,
        risk_free_rate,
        initial_capital,
        commission,
        alpha,
        threshold,
        order,
        order_type,
        short,
        short_fee,
        trailing_take_profit,
        take_profit,
        trailing_stop_loss,
        stop_loss,
        leverage,
        miss_rate,
        show_initial_configuration,
        show_tables,
        show_charts,
        show_time,
        precision_point,
    )
    return metrics[metric]


def optimize_backtest(
    metric_optimized: str,
    take_profit_values: list,
    stop_loss_values: list,
    leverage_values: list,
    iteration: int,
    best_n: int = 3,
    verbose: bool = True,
    show_results: bool = True,
):
    start = time.time()
    combinations = np.array(
        np.meshgrid(take_profit_values, stop_loss_values, leverage_values)
    ).T.reshape(-1, 3)
    my_bar = st.progress(0)
    results = {}
    t = st.empty()
    length = len(combinations)
    for combination_count, combination in enumerate(combinations, start=1):
        key = tuple(combination)
        results[key] = metric_optimization(
            metric_optimized, key[0], key[1], key[2]
        )
        for _ in range(iteration - 1):
            results[key] += metric_optimization(
                metric_optimized, key[0], key[1], key[2]
            )
        results[key] = results[key] / (iteration)
        my_bar.progress(combination_count / length)
        if verbose:
            t.markdown(
                f"{combination_count}/{len(combinations)} (%{round(combination_count/len(combinations)*100,2)}) of combinations were tested. {second_2_minute_converter(time.time()-start)} passed."
            )
    sorted_dict = dict(
        sorted(results.items(), key=lambda item: item[1], reverse=True)
    )
    st.markdown("<br> <br>", unsafe_allow_html=True)
    if show_results:
        tf = []
        sl = []
        lv = []
        om = []
        count = 0
        for key, value in sorted_dict.items():
            tf.append(key[0])
            sl.append(key[1])
            lv.append(key[2])
            om.append([round(value, 2)])
            count += 1
            if count >= best_n:
                break
        fig = make_subplots(
            rows=1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            specs=[[{"type": "table"}]],
        )

        fig.add_trace(
            go.Table(
                columnwidth=[1, 1, 1, 1],
                header=dict(
                    values=[
                        "Take Profit",
                        "Stop Loss",
                        "Leverage",
                        f"{metric_optimized}",
                    ],
                    line_color="darkslategray",
                    fill_color="lightskyblue",
                    align="center",
                ),
                cells=dict(
                    values=[tf, sl, lv, om],
                    line_color="darkslategray",
                    fill_color="lightcyan",
                    align="center",
                ),
            ),
            row=1,
            col=1,
        )
        fig.update_layout(
            width=1000,
            height=1000,
            title_text="<span style='font-size: 30px;'><b>OPTIMIZATION RESULTS</b></span>",
            title_x=0.5,
        )
        st.plotly_chart(fig, use_container_width=True)
    end = time.time()
    if verbose:
        st.write(
            f"\nOptimization was completed in {second_2_minute_converter(end-start)} with {len(combinations)} combinations and {iteration} iteration for each combination.\n"
        )
    return {key: sorted_dict[key] for key in list(sorted_dict.keys())[:best_n]}
