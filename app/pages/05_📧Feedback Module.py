import streamlit as st
from configuration import (
    add_bg_from_local,
    configure_authors,
    configure_page,
    local_css,
)


def main():
    configure_page()
    configure_authors()
    add_bg_from_local("data/background.png", "data/bot.png")
    local_css("style/style.css")

    st.markdown(
        "<h1 style='text-align: center; color: black; font-size: 65px;'> 📧 Feedback Module </h1> <br>",
        unsafe_allow_html=True,
    )

    feedback_message = '<p style="font-family:Arial Black; font-size: 30px;" align="center"> \
    You can use the text area below to send your feedback about the app to the developer. Thanks! </p>'
    st.markdown(feedback_message, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    contact_form = """
    <form action="https://formsubmit.co/kuantum21fizik@gmail.com" method="POST" align="center">
        <input type="hidden" name="_captcha" value="false">
        <input type="hidden" name="_subject" value="Trading Bot Feedback!">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here"></textarea>
        <button type="submit">Send</button>
        <input type="hidden" name="_next" value="https://alpha-tradingbot.streamlit.app/">
    </form>
    """
    _, center_col, _ = st.columns([1, 3, 1])
    center_col.markdown(contact_form, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
