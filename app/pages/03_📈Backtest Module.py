import create_backtest as cb
import numpy as np
import streamlit as st

pass
from configuration import add_bg_from_local, configure_authors, configure_page

configure_page()
configure_authors()
add_bg_from_local("data/background.png", "data/bot.png")

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


st.markdown(
    """
    <style>
        div[data-testid="column"]:nth-of-type(1)
        {
            text-align: center;
        }

        div[data-testid="column"]:nth-of-type(2)
        {
            text-align: center;
        }
        div[data-testid="column"]:nth-of-type(3)
        {
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    "<h1 style='text-align: center; color: black; font-size: 65px;'> 📈 Backtest Module </h1> <br> <br>",
    unsafe_allow_html=True,
)

style = "<style>.row-widget.stButton {text-align: center;}</style>"
st.markdown(style, unsafe_allow_html=True)
st.markdown("<br> <br>", unsafe_allow_html=True)
if (
    st.session_state["data"] is None
    or st.session_state["signals"] is None
    or st.session_state["ticker"] == ""
):
    st.error("Please get the data and create the strategy first.")
else:
    ticker = st.session_state["ticker"]
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        show_initial_configuration = st.checkbox("Show initial configuration")
    with col2:
        show_tables = st.checkbox("Show tables")
    with col3:
        show_charts = st.checkbox("Show charts")
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("See extra configuration for backtest"):
        col1, col2 = st.columns([1, 1])
        with col1:
            benchmark_ticker = st.text_input(
                "Enter the benchmark ticker (Default is SPY)", "SPY"
            )
            st.session_state["backtest_configuration"][
                "benchmark_ticker"
            ] = benchmark_ticker
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
            buy_label = st.number_input("Enter the buy label", value=1, step=1)
            st.session_state["backtest_configuration"]["buy_label"] = buy_label
        with col3:
            sell_label = st.number_input(
                "Enter the sell label", value=2, step=1
            )
            st.session_state["backtest_configuration"][
                "sell_label"
            ] = sell_label
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            initial_capital = st.number_input(
                "Enter the initial capital", value=1000.0, step=1000.0
            )
            st.session_state["backtest_configuration"][
                "initial_capital"
            ] = initial_capital
        with col2:
            risk_free_rate = (
                st.number_input(
                    "Enter the risk free rate", value=0.01, step=0.01
                )
                / 252
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
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            alpha = st.number_input("Enter the alpha", value=0.05, step=0.01)
            st.session_state["backtest_configuration"]["alpha"] = alpha
        with col2:
            threshold = st.number_input(
                "Enter the threshold", value=0.0, step=1.0
            )
            st.session_state["backtest_configuration"]["threshold"] = threshold
        with col3:
            order = st.number_input("Enter the order", value=1.0, step=1.0)
            st.session_state["backtest_configuration"]["order"] = order
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
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
            st.session_state["backtest_configuration"]["miss_rate"] = miss_rate
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
        col1, col2, col3 = st.columns([1, 1, 1])
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
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            short_fee = st.number_input(
                "Enter the short fee (Default is 1)", value=1.0, step=1.0
            )
            st.session_state["backtest_configuration"]["short_fee"] = short_fee
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
            st.session_state["backtest_configuration"]["stop_loss"] = stop_loss
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            leverage = st.number_input(
                "Enter the leverage multiplier (Default is 1)",
                value=1.0,
                step=1.0,
            )
            st.session_state["backtest_configuration"]["leverage"] = leverage
        with col2:
            pass
        with col3:
            pass

        st.markdown("<br>", unsafe_allow_html=True)
    initial_conf_df, charts_dict, portfolio, benchmark = [None] * 4
    if st.button("Run the backtest"):
        st.session_state.run_backtest_button_clicked = True
        st.session_state["backtest_configuration_ready"] = True
        (
            st.session_state.portfolio,
            st.session_state.benchmark,
            charts_dict_params,
        ) = cb.financial_evaluation(
            hold_label,
            buy_label,
            sell_label,
            ticker,
            benchmark_ticker,
            data,
            signals,
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
            True,
            precision_point,
        )
        st.session_state.initial_conf_df = cb.plot_init(
            ticker,
            benchmark_ticker,
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
    if (
        st.session_state.initial_conf_df is not None
        and st.session_state.charts_dict is not None
        and st.session_state.portfolio is not None
        and st.session_state.benchmark is not None
        and st.session_state.run_backtest_button_clicked
    ):
        st.markdown("<br><br>", unsafe_allow_html=True)
        if show_initial_configuration:
            _, icd_col, _ = st.columns([2, 3, 2])
            icd_col.dataframe(st.session_state.initial_conf_df, width=500)

        strategy_returns = (
            st.session_state.portfolio["Value"].pct_change().dropna()
        )
        benchmark_returns = (
            st.session_state.benchmark["Close"].pct_change().dropna()
        )

        metrics_dict, metrics_df = cb.qs_metrics(
            strategy_returns, benchmark_returns, risk_free_rate=1
        )
        plots_dict = cb.qs_plots(strategy_returns, benchmark_returns)

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
                st.session_state.charts_dict,
                st.session_state.last_strategy,
            ),
        )

        st.markdown("<br><br>", unsafe_allow_html=True)
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
