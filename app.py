import streamlit as st
from PIL import Image
st.set_page_config(
    page_title="AI Assistance for Sign Language",
    page_icon="🗣"
)
st.sidebar.success("Select a page above")

st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>AI Assistance for Sign Language</h1>", unsafe_allow_html=True)
st.markdown("###") #To add space in between
st.subheader("An AI Assistance Tool for Sign Language that bridges the communication gap between the deaf-mute and those who can speak.")


# st.markdown("Way to our Repository 👉 [![Star](https://img.shields.io/github/stars/harikrish-s/sign-to-speech-main.svg?logo=github&style=social)](https://github.com/harikrish-s/Sign-Language-Recognition)")