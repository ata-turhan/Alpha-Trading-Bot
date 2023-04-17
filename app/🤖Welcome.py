import streamlit as st
from configuration import add_bg_from_local, configure_authors


def main():
    st.set_page_config(
        page_title="🤖 Trading Bot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/olympian-21",
            "Report a bug": "https://alpha-tradingbot.streamlit.app/Feedback_Module",
            "About": "This is a trading bot that can be used for retrieving financial data, \
            creating trading strategies, backtesting the strategies, and optimizing the backtests. \
            Please, give us all the helpful feedback!",
        },
    )

    add_bg_from_local("data/background.png", "data/bot.png")

    configure_authors()

    st.markdown(  # Combined text and styling into one line for readability
        "<h1 style='text-align: center; color: black; font-size: 75px;'> 🤖 Fully-Fledged Trading Bot </h1> \
        <br> <br> <br>",
        unsafe_allow_html=True,
    )

    welcome_message = '<p style="font-family:Arial Black; font-size: 20px;" align="center"> \
    This is a web app that allows you to <br> \
        • Gather financial information from any financial market.  <br> \
        • Create trading strategies with various methods <br> \
        • Thoroughly backtest these strategies. <br> \
        • Optimize these backtests without overfitting <br> \
                                               <br> \
        You can use the modules on the sidebar to navigate in the web app. \
    </p>'
    st.markdown(welcome_message, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
