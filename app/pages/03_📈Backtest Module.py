import create_backtest as cb
import numpy as np
import streamlit as st
from configuration import (
    add_bg_from_local,
    configure_authors,
    configure_page,
    local_css,
)


def set_session_variables():
    if "data" not in st.session_state:
        st.session_state.data = None
    if "signals" not in st.session_state:
        st.session_state.signals = None
    if "ticker" not in st.session_state:
        st.session_state.ticker = ""
    if "backtest_configuration_ready" not in st.session_state:
        st.session_state.backtest_configuration_ready = False
    if "backtest_configuration" not in st.session_state:
        st.session_state.backtest_configuration = {}
    if "run_backtest_button_clicked" not in st.session_state:
        st.session_state.run_backtest_button_clicked = False
    if "initial_conf_df" not in st.session_state:
        st.session_state.initial_conf_df = None
    if "charts_dict" not in st.session_state:
        st.session_state.charts_dict = None
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = None
    if "benchmark" not in st.session_state:
        st.session_state.benchmark = None
    if "strategy_returns" not in st.session_state:
        st.session_state.strategy_returns = None
    if "benchmark_returns" not in st.session_state:
        st.session_state.benchmark_returns = None
    if "metrics_dict" not in st.session_state:
        st.session_state.metrics_dict = None
    if "metrics_df" not in st.session_state:
        st.session_state.metrics_df = None
    if "plots_dict" not in st.session_state:
        st.session_state.plots_dict = None


def determine_benchmark(market):
    if market == "Stock":
        st.session_state.backtest_configuration["benchmark_ticker"] = "AAPL"
    elif market == "ETF":
        st.session_state.backtest_configuration["benchmark_ticker"] = "SPY"
    elif market == "Forex":
        st.session_state.backtest_configuration[
            "benchmark_ticker"
        ] = "EURUSD=X"
    elif market == "Crypto":
        st.session_state.backtest_configuration["benchmark_ticker"] = "BTC-USD"


def main():
    set_session_variables()
    configure_page()
    configure_authors()
    add_bg_from_local("data/background.png", "data/bot.png")
    local_css("style/style.css")

    st.markdown(
        "<h1 style='text-align: center; color: black; font-size: 65px;'> 📈 Backtest Module </h1> <br> <br>",
        unsafe_allow_html=True,
    )

    st.markdown("<br> <br>", unsafe_allow_html=True)
    if (
        st.session_state["data"] is None
        or st.session_state["signals"] is None
        or st.session_state["ticker"] == ""
    ):
        st.error("Please get the data and create the strategy first.")
    else:
        ticker = st.session_state["ticker"]
        determine_benchmark(st.session_state.market)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            show_initial_configuration = st.checkbox(
                "Show initial configuration"
            )
        with col2:
            show_tables = st.checkbox("Show tables")
        with col3:
            show_charts = st.checkbox("Show charts")
        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("See extra configuration for backtest"):
            data = st.session_state["data"]
            signals = np.array(st.session_state["signals"])
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                hold_label = st.number_input(
                    "Enter the hold label", value=0, step=1
                )
                st.session_state["backtest_configuration"][
                    "hold_label"
                ] = hold_label
            with col2:
                buy_label = st.number_input(
                    "Enter the buy label", value=1, step=1
                )
                st.session_state["backtest_configuration"][
                    "buy_label"
                ] = buy_label
            with col3:
                sell_label = st.number_input(
                    "Enter the sell label", value=2, step=1
                )
                st.session_state["backtest_configuration"][
                    "sell_label"
                ] = sell_label
            st.markdown("<br>", unsafe_allow_html=True)
            with col1:
                initial_capital = st.number_input(
                    "Enter the initial capital", value=1000.0, step=1000.0
                )
                st.session_state["backtest_configuration"][
                    "initial_capital"
                ] = initial_capital
            with col2:
                risk_free_rate = st.number_input(
                    "Enter the risk free rate", value=0.03, step=0.01
                )
                st.session_state["backtest_configuration"][
                    "risk_free_rate"
                ] = risk_free_rate
            with col3:
                commission = st.number_input(
                    "Enter the commission rate", value=1.0, step=1.0
                )
                st.session_state["backtest_configuration"][
                    "commission"
                ] = commission
            st.markdown("<br>", unsafe_allow_html=True)
            with col1:
                alpha = st.number_input(
                    "Enter the alpha", value=0.05, step=0.01
                )
                st.session_state["backtest_configuration"]["alpha"] = alpha
            with col2:
                threshold = st.number_input(
                    "Enter the threshold", value=0.0, step=1.0
                )
                st.session_state["backtest_configuration"][
                    "threshold"
                ] = threshold
            with col3:
                order = st.number_input("Enter the order", value=1.0, step=1.0)
                st.session_state["backtest_configuration"]["order"] = order
            st.markdown("<br>", unsafe_allow_html=True)
            with col1:
                order_type = st.selectbox(
                    "Select the order type", ["market", "limit"]
                )
                st.session_state["backtest_configuration"][
                    "order_type"
                ] = order_type
            with col2:
                miss_rate = st.number_input(
                    "Enter the trade miss rate", value=10, step=1
                )
                st.session_state["backtest_configuration"][
                    "miss_rate"
                ] = miss_rate
            with col3:
                precision_point = st.number_input(
                    "Enter the precision point to show decimal numbers",
                    value=3,
                    step=1,
                )
                st.session_state["backtest_configuration"][
                    "precision_point"
                ] = precision_point
            st.markdown("<br>", unsafe_allow_html=True)
            with col1:
                short = st.checkbox("Use short positions")
                st.session_state["backtest_configuration"]["short"] = short
            with col2:
                trailing_take_profit = st.checkbox("Use trailing take profit")
                st.session_state["backtest_configuration"][
                    "trailing_take_profit"
                ] = trailing_take_profit
            with col3:
                trailing_stop_loss = st.checkbox("Use trailing stop loss")
                st.session_state["backtest_configuration"][
                    "trailing_stop_loss"
                ] = trailing_stop_loss
            st.markdown("<br>", unsafe_allow_html=True)
            with col1:
                short_fee = st.number_input(
                    "Enter the short fee (Default is 1)", value=1.0, step=1.0
                )
                st.session_state["backtest_configuration"][
                    "short_fee"
                ] = short_fee
            with col2:
                take_profit = st.number_input(
                    "Enter the take profit percentage (Default is 10)",
                    value=5.0,
                    step=1.0,
                )
                st.session_state["backtest_configuration"][
                    "take_profit"
                ] = take_profit
            with col3:
                stop_loss = st.number_input(
                    "Enter the stop loss percentage (Default is 10)",
                    value=5.0,
                    step=1.0,
                )
                st.session_state["backtest_configuration"][
                    "stop_loss"
                ] = stop_loss
            st.markdown("<br>", unsafe_allow_html=True)
            with col2:
                leverage = st.number_input(
                    "Enter the leverage multiplier (Default is 1)",
                    value=1.0,
                    step=1.0,
                )
                st.session_state["backtest_configuration"][
                    "leverage"
                ] = leverage
        initial_conf_df, charts_dict, portfolio, benchmark = [None] * 4
        if st.button("Run the backtest"):
            st.session_state.run_backtest_button_clicked = True
            st.session_state["backtest_configuration_ready"] = True
            (
                st.session_state.portfolio,
                st.session_state.benchmark,
                charts_dict_params,
            ) = cb.financial_evaluation(
                ohlcv=data,
                predictions=signals,
                show_initial_configuration=show_initial_configuration,
                show_tables=show_tables,
                show_charts=show_charts,
                show_time=True,
                **st.session_state["backtest_configuration"],
            )
            st.session_state.initial_conf_df = cb.plot_init(
                ticker,
                st.session_state["backtest_configuration"]["benchmark_ticker"],
                risk_free_rate,
                data.index[0],
                data.index[-1],
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
            st.session_state.charts_dict = cb.plot_charts(*charts_dict_params)

            st.session_state.strategy_returns = (
                st.session_state.portfolio["Value"].pct_change().dropna()
            )
            st.session_state.benchmark_returns = (
                st.session_state.benchmark["Close"].pct_change().dropna()
            )

            (
                st.session_state.metrics_dict,
                st.session_state.metrics_df,
            ) = cb.qs_metrics(
                st.session_state.strategy_returns,
                st.session_state.benchmark_returns,
                risk_free_rate=0.03,
            )
            st.session_state.plots_dict = cb.qs_plots(
                st.session_state.strategy_returns,
                st.session_state.benchmark_returns,
            )

        if (
            st.session_state.initial_conf_df is not None
            and st.session_state.charts_dict is not None
            and st.session_state.portfolio is not None
            and st.session_state.benchmark is not None
            and st.session_state.run_backtest_button_clicked
            and st.session_state.strategy_returns is not None
            and st.session_state.benchmark_returns is not None
            and st.session_state.metrics_dict is not None
            and st.session_state.plots_dict is not None
        ):
            initial_conf_df = st.session_state.initial_conf_df
            charts_dict = (st.session_state.charts_dict,)
            last_strategy = (st.session_state.last_strategy,)
            metrics_dict = st.session_state.metrics_dict
            metrics_df = st.session_state.metrics_df
            plots_dict = st.session_state.plots_dict

            col1, col2 = st.columns(
                [
                    1,
                    1,
                ]
            )

            col1.download_button(
                "Download the metrics of the backtest",
                metrics_df.to_html().encode("utf-8"),
                "Metrics.html",
            )
            col2.button(
                "Download the plots of the backtest",
                on_click=cb.generate_qs_plots_report,
                args=(
                    plots_dict,
                    charts_dict,
                    last_strategy,
                ),
            )

            st.markdown("<br>", unsafe_allow_html=True)
            if show_initial_configuration:
                _, icd_col, _ = st.columns([2, 3, 2])
                icd_col.dataframe(initial_conf_df, width=500)

            st.markdown("<br>", unsafe_allow_html=True)
            if show_tables:
                col1, col2, col3 = st.columns([1, 1, 1])
                col1.subheader("Returns")
                col1.dataframe(metrics_dict["returns"], width=500)
                col2.subheader("Ratios")
                col2.dataframe(metrics_dict["ratios"], width=500)
                col3.subheader("Risks")
                col3.dataframe(metrics_dict["risks"], width=500)

                col1, col2 = st.columns([1, 1])
                col1.subheader("Trade Counts")
                col1.dataframe(metrics_dict["counts"], width=500)
                col2.subheader("Extremums")
                col2.dataframe(metrics_dict["extremums"], width=500)

                # st.dataframe(data)
                # st.dataframe(bench)
                # st.dataframe(returns)
                # st.dataframe(bench_returns)

            if show_charts:
                st.plotly_chart(
                    st.session_state.charts_dict["transactions"],
                    use_container_width=True,
                )
                st.plotly_chart(
                    st.session_state.charts_dict["portfolio_value"],
                    use_container_width=True,
                )

                col1, col2 = st.columns([1, 1])
                col1.write(plots_dict["snapshot"])
                col2.write(plots_dict["heatmap"])

                col1, col2 = st.columns([1, 1])
                col1.write(plots_dict["normal_returns"])
                col2.write(plots_dict["log_returns"])

                col1, col2 = st.columns([1, 1])
                col1.write(plots_dict["histogram"])
                col2.write(plots_dict["distribution"])

                col1, col2 = st.columns([1, 1])
                col1.write(plots_dict["rolling_beta"])
                col2.write(plots_dict["rolling_volatility"])

                col1, col2 = st.columns([1, 1])
                col1.write(plots_dict["rolling_sharpe"])
                col2.write(plots_dict["rolling_sortino"])

                col1, col2 = st.columns([1, 1])
                col1.write(plots_dict["drawdowns_periods"])
                col2.write(plots_dict["drawdown"])

                # if st.button("Download the plots of the backtest"):


if __name__ == "__main__":
    main()
