import pandas as pd
import streamlit as st

# title
html_title = '<h1 align="center"> <b>◍ Wafer Fault Detection ◍ </b></h1>'
st.markdown(html_title, unsafe_allow_html=True)
st.markdown('#')

# introduction
st.markdown('<p align="justify"> This is an autonomous semi-supervised machine learning application to detect the quality of electronic wafers based on the inputs from various sensors. A wafer is a thin slice of semiconductor used for the fabrication of integrated circuits and, in photovoltaics, to manufacture solar cells. </p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,3,1])
col2.image('demo\Picture1.png', caption='electronic wafer with built in circuits')

# data description
st.markdown('##')
st.markdown('<p align="justify"> The training data is in a set of files (batches) and each file contains a wafer name column, 590 columns of different sensor values, and a wafer quality column (Good/Bad). The prediction data should also be in batches and each must contain a wafer name column and 590 columns of different sensor values. </p>', unsafe_allow_html=True)

st.markdown('<p align="center"> sample of a training file </p>', unsafe_allow_html=True)
df_s = pd.read_csv('demo\Wafer_07012020_000000.csv')
st.dataframe(df_s)


# application architecture
st.markdown('##')
st.markdown('<p align="justify"> The project architecture is made up of two main pipelines (training and prediction). The training pipeline contains three stages namely data ingestion, data preprocessing, and model development, while the prediction pipeline contains data ingestion, data preprocessing, and prediction. </p>', unsafe_allow_html=True)

st.image('demo\Picture2.png', caption='project architecture')


