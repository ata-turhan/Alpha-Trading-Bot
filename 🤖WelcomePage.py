#Import the required Libraries
import streamlit as st
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
    """,
    unsafe_allow_html=True
    )


def main():
    st.set_page_config(page_title='🤖 Trading Bot', page_icon='🤖', layout="wide", initial_sidebar_state="expanded",
                      menu_items={
                                'Get Help': 'https://github.com/fotino21',
                                'Report a bug': None,
                                'About': "This is a trading bot which can be used for retrival of financial data, \
                                creating trading strategies, backtesting the strategies and optimizing the strategies. \
                                Please, give us all the helpful feedbacks!"
    })

    add_bg_from_local('data/background.png') 

    for _ in range(20):
        st.sidebar.text("\n")
    st.sidebar.write('Developed by Ata Turhan')
    st.sidebar.write('Contact at ataturhan21@gmail.com')

    """
    [![Follow](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ata-turhan-555b5b160/)
    """

    st.markdown("<h1 style='text-align: center; color: black;'> 🤖 Fully-Fledged Trading  Bot </h1> <br> <br>", unsafe_allow_html=True)

    # Add a title and intro text
    st.write("This is a web app that allows you to")
    st.write("• get financial data in any financial market")
    st.write("• create trading strategies with various methods")
    st.write("• backtest these strategies extensively")
    st.write("\n\n")
    st.write("You can use the modules on the sidebar to navigate in the web app.")

if __name__ == "__main__":
    main()

