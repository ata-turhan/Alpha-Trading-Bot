import streamlit as st
from configuration import add_bg_from_local, configure_authors, configure_page

configure_page()
configure_authors()
add_bg_from_local("data/background.png", "data/bot.png")

st.markdown(
    "<h1 style='text-align: center; color: black;'> 📧 Feedback Module </h1> <br> <br>",
    unsafe_allow_html=True,
)
st.markdown("<br> <br>", unsafe_allow_html=True)
st.header(
    "You can use the text area below to send your feedback about the app to the developer. Thanks!"
)

contact_form = """
<form action="https://formsubmit.co/kuantum21fizik@gmail.com" method="POST">
     <input type="hidden" name="_captcha" value="false">
     <input type="hidden" name="_subject" value="Trading Bot Feedback!">
     <input type="text" name="name" placeholder="Your name" required>
     <input type="email" name="email" placeholder="Your email" required>
     <textarea name="message" placeholder="Your message here"></textarea>
     <button type="submit">Send</button>
     <input type="hidden" name="_next" value="https://trading-bot.streamlit.app/">
</form>
"""

st.markdown(contact_form, unsafe_allow_html=True)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")
