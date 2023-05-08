import numpy as np
import optimization as op
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


def main():
    set_session_variables()
    configure_page()
    configure_authors()
    add_bg_from_local("data/background.png", "data/bot.png")
    local_css("style/style.css")

    st.markdown(
        "<h1 style='text-align: center; color: black; font-size: 65px;'> ⚡Optimization Module </h1> <br> ",
        unsafe_allow_html=True,
    )

    if (
        st.session_state["data"] is None
        or st.session_state["signals"] is None
        or st.session_state["ticker"] == ""
        or st.session_state["backtest_configuration_ready"] == False
    ):
        st.error(
            "Please get the data, create the strategy and backtest the strategy first."
        )
    else:
        _, center_col, _ = st.columns([1, 2, 1])
        metric_optimized = center_col.selectbox(
            "Please select the backtest metrics to optimize:",
            [
                "<Select>",
                "Cumulative Return",
                "Sharpe",
                "Sortino",
                "Omega",
                "Treynor Ratio",
            ],
        )
        st.markdown("<br>", unsafe_allow_html=True)
        take_profit_ranges = center_col.slider(
            "Select a range for take profit values", 0, 50, (5, 25)
        )
        take_profit_values = list(
            range(take_profit_ranges[0], take_profit_ranges[1])
        )
        st.markdown("<br>", unsafe_allow_html=True)
        stop_loss_ranges = center_col.slider(
            "Select a range for stop loss values", 0, 50, (5, 25)
        )
        stop_loss_values = list(
            range(stop_loss_ranges[0], stop_loss_ranges[1])
        )
        st.markdown("<br>", unsafe_allow_html=True)
        leverage_ranges = center_col.slider(
            "Select a range for leverage values", 1, 20, (1, 5)
        )
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = center_col.columns([1, 1])
        population_size = int(
            col1.number_input(
                "Choose the population size.",
                value=15,
                max_value=100,
                min_value=10,
                step=5,
            )
        )
        mutation_probability = col2.number_input(
            "Choose the mutation probability.",
            value=0.1,
            max_value=1.0,
            min_value=0.0,
            step=0.1,
        )

        _, col2, _ = center_col.columns([1, 4, 1])
        with col2:
            iteration = int(
                st.number_input(
                    "Choose the iteration number to try optimized values.",
                    value=0,
                )
            )

        _, col2, _ = center_col.columns([1, 3, 1])
        col2.markdown("<br>", unsafe_allow_html=True)
        if col2.button("Run the optimization"):
            if metric_optimized == "<Select>" or iteration == 0:
                center_col.error("Please fill all the required fields.")
            else:
                op.optimize(
                    col2,
                    ohlcv=st.session_state.data,
                    predictions=np.array(st.session_state.signals),
                    metric_optimized=metric_optimized,
                    take_profit_values=take_profit_ranges,
                    stop_loss_values=stop_loss_ranges,
                    leverage_values=leverage_ranges,
                    population_size=population_size,
                    mutation_probability=mutation_probability,
                    iteration=iteration,
                )
                center_col.success("Optimization is successful!")
                center_col.balloons()


if __name__ == "__main__":
    main()
