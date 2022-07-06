import streamlit as st
import pandas as pd
import plotly.express as px
import cv2
from application_functions import predict, blend_images
import os
from PIL import Image

st.set_page_config(layout="wide")
st.markdown('<style>body{background-color: White;}</style>',unsafe_allow_html=True)


show_labels = True

st.title('Semantic Segmentation of Historial Documents')

expander_bar = st.expander("About")
expander_bar.markdown(
    """
* **Semantic Segmention of Historical Documents using Deep Learning Architectures**
* Prediction of Semantic Segmentation masks for images of the **[Eparchos Dataset](https://zenodo.org/record/4095301#.YsV2rHZBzDc)**.
* Deep Learning Architectures Used: **[dhSegment](https://dhsegment.readthedocs.io/en/latest/intro/intro.html)**, **[U-Net](https://arxiv.org/abs/1505.04597)**, **[VGG16](https://keras.io/api/applications/vgg/)**. 
* This consists a demo of my [**Graduate Thesis**](https://drive.google.com/file/d/1MoOnG4wPs2h1XBP2vNGyT-Y3iS9OFyBp/view?usp=sharing).
"""
)


settings_column, input_column, output_column, blended_column = st.columns((1, 2, 2, 2))

settings_column.header("**Settings**")
input_column.header("**Input Image**")
output_column.header("**Predicted Image**")
blended_column.header("**Blended Image**")

input_images_dir = "images"
list_filenames = [filename for filename in sorted(os.listdir(input_images_dir))]

task_list = ["basic", "advanced"]
task_sel = settings_column.radio("Task", options = task_list)

if show_labels:
    if task_sel == "basic":
        labels_basic = Image.open("labels_basic.jpg")
        settings_column.image(labels_basic, caption="List of Classes Labels")
    elif task_sel == "advanced":
        labels_advanced = Image.open("labels_advanced.jpg")
        settings_column.image(labels_advanced, caption="List of Classes Labels")

model_list = ["dhSegment", "U-Net + VGG16", "U-Net"]
model_sel = settings_column.radio("Model Architecture", options = model_list)

if task_sel == "basic":

    if model_sel == "dhSegment":
        model_dir = "model_basic_resnet50\export"
    elif model_sel == "U-Net + VGG16":
        model_dir = "model_basic_vgg16_weights\export"
    elif model_sel == "U-Net":
        model_dir = "model_basic_unet\export"

elif task_sel == "advanced":

    if model_sel == "dhSegment":
        model_dir = "model_advanced_resnet50\export"
    elif model_sel == "U-Net + VGG16":
        model_dir = "model_advanced_vgg16_weights\export"
    elif model_sel == "U-Net":
        model_dir = "model_advanced_unet\export"

input_image_path = settings_column.selectbox("Input Image", options = list_filenames)
image_path = f"images/{input_image_path}"

if image_path is not None:

    input_image = cv2.imread(image_path, 1)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    
    input_column.image(input_image)

    output_image = predict(model_dir, image_path, task_sel)
    blended_image = blend_images(input_image, output_image)

    output_column.image(output_image)
    blended_column.image(blended_image)

else:
    input_column.header("Please Choose a file for the input image.")


st.markdown('Created by [**Anastasios Tzelepakis**](https://www.linkedin.com/in/anastasios-tzelepakis/).')