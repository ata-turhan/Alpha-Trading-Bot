import base64

import streamlit as st


def configure_authors():
    st.sidebar.markdown("<br> " * 12, unsafe_allow_html=True)
    # Added to separate contact info from other sidebar elements

    st.sidebar.write(
        "Developed by Ata Turhan & Orçun Demir"
    )  # Moved contact info to the top of the sidebar

    st.sidebar.write(
        "Contact at ataturhan21@gmail.com"
    )  # Moved contact info to the top of the sidebar

    st.sidebar.markdown(
        """  [![Follow](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ata-turhan-555b5b160/) """
    )


def add_bg_from_local(background_file, sidebar_background_file):
    with open(background_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    with open(sidebar_background_file, "rb") as image_file:
        sidebar_encoded_string = base64.b64encode(image_file.read())

    page = f"""<style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string.decode()});
            background-size: cover;
        }}

        section[data-testid="stSidebar"] div[class="css-6qob1r e1fqkh3o3"] {{
            background-image: url(data:image/png;base64,{sidebar_encoded_string.decode()});
            background-size: 400px 800px;
        }}
    </style>"""

    st.markdown(page, unsafe_allow_html=True)
