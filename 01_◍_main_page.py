import streamlit as st
from PIL import Image

# title
html_title = '<h1 align="center"> <b>◍ Wafer Fault Prediction Project ◍ </b></h1>'
st.markdown(html_title, unsafe_allow_html=True)
st.markdown('#')

image = Image.open('demo\Picture1.png')

st.image(image)
