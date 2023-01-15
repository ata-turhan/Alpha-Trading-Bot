# Import the required Libraries
import base64

import streamlit as st


def add_bg_from_local(background_file, sidebar_background_file):
    with open(background_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    with open(sidebar_background_file, "rb") as image_file:
        sidebar_encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    section[data-testid="stSidebar"] div[class="css-6qob1r e1fqkh3o3"]
    {{
        background-image: url(data:image/{"png"};base64,{sidebar_encoded_string.decode()});
        background-size: 400px 800px
    }}
    """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="🤖 Trading Bot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/fotino21",
            "Report a bug": None,
            "About": "This is a trading bot which can be used for retrival of financial data, \
                                creating trading strategies, backtesting the strategies and optimizing the strategies. \
                                Please, give us all the helpful feedbacks!",
        },
    )

    add_bg_from_local("data/background.png", "data/bot.png")

    for _ in range(18):
        st.sidebar.text("\n")
    st.sidebar.write("Developed by Ata Turhan")
    st.sidebar.write("Contact at ataturhan21@gmail.com")

    """
    [![Follow](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ata-turhan-555b5b160/)
    """

    # st.markdown("""
    # <style>
    # section[data-testid="stSidebar"] div[class="css-6qob1r e1fqkh3o3"]
    # {background-image: linear-gradient(#8993ab,#8993ab);color: white}
    # </style>
    # """,
    # unsafe_allow_html=True)

    st.markdown(
        "<h1 style='text-align: center; color: black;'> 🤖 Fully-Fledged Trading  Bot </h1> <br> <br>",
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
