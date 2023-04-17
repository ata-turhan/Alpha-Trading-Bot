# Import the required Libraries
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
            "Report a bug": None,
            "About": "This is a trading bot which can be used for retrival of financial data, creating trading strategies, backtesting the strategies and optimizing the strategies. Please, give us all the helpful feedbacks!",
        },
    )

    add_bg_from_local("data/background.png", "data/bot.png")

    configure_authors()

    st.markdown(  # Combined text and styling into one line for readability
        "<h1 style='text-align: center; color: black;'> 🤖 Fully-Fledged Trading Bot </h1> <br> <br>",
        unsafe_allow_html=True,
    )

    welcome_message = '<p style="font-family:Arial; font-size: 26px;"> \
    This is a web app that allows you to <br> \
        • get financial data in any financial market <br> \
        • create trading strategies with various methods <br> \
        • backtest these strategies extensively <br> \
                                               <br> \
        You can use the modules on the sidebar to navigate in the web app. \
    </p>'
    st.markdown(welcome_message, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
