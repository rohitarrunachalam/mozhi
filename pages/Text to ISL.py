import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Text to ISL",
)

st.header("Text to ISL Conversion")
data = st.text_input("Enter the text you want to convert :")
images = []
for i in data:
    if (i.isnumeric() == True):
        image = Image.open(
        r"Dataset_to_print/"+i+"/"+"1.jpg")
        images.append(image)
    elif (i.isalpha() == True):
        j = i.upper()
        del i
        i = j
        image = Image.open(
        r"Dataset_to_print/"+i+"/"+"1.jpg")
        images.append(image)
    elif(i.isspace() == True):
        image = Image.open(
        r"Dataset_to_print/"+"space.jpg")
        images.append(image)
st.image(images)
