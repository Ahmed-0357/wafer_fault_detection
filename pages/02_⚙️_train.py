import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from st_app_funcs import *

# title
html_title = '<h1 align="center"> <b>⚙️ Model Training ⚙️</b></h1>'
st.markdown(html_title, unsafe_allow_html=True)
st.markdown('#')



button1 = st.button('Check 1')

if st.session_state.get('button') != True:

    st.session_state['button'] = button1

if st.session_state['button'] == True:

    st.write("button1 is True")

    if st.button('Check 2'):

        st.write("Hello, it's working")

        st.session_state['button'] = False

        st.checkbox('Reload')



# data ingestion
st.markdown('## Data Ingestion')
# open training DSA
with open('DSA_schema\schema_training.json') as f:
    help_c = json.load(f)
uploaded_files = st.file_uploader('Upload batch files', type='csv',
                                accept_multiple_files=True, help='the ideal file should follow this DSA format: '+str(help_c))
 
df_i, df_p = None, None
if len(uploaded_files) != 0:
    with st.spinner('ingesting files...'):
        good_files = save_ingest_data(uploaded_files)
        st.success('files has been ingested successfully')
    st.write('files matching DSA')
    st.write(good_files)
    df_i = pd.read_csv(r'training_data\train.csv')
else: # sample dataset 
    sample_data = st.checkbox(
    'use sample dataset')
    if sample_data:
        df_i = pd.read_csv(r'training_data\train.csv')

if df_i is not None:
    df_i.iloc[:,-1] = df_i.iloc[:,-1].replace([-1], 0)
    df_i.iloc[:,-1] = df_i.iloc[:,-1].astype('str')
    st.write('Sample dataset')
    st.dataframe(df_i)
    fig1 = px.pie(df_i,names=df_i.columns[-1], title='output column')
    st.plotly_chart(fig1)
            

    # data ingestion
    st.markdown('## Data Preprocessing')
    if st.button('click here', help='data preprocessing includes data cleaning, train-validation-test split, missing data imputation, features scaling and pca decomposition'):
        with st.spinner('data preprocessing...'):
            data_process()
            st.success('data has been preprocessed successfully')
            df_p = pd.read_csv(r'training_data\train_split.csv')
            
if df_p is not None:
    st.write('preprocessed training split')
    df_p.iloc[:,-1] = df_p.iloc[:,-1].astype('str')
    st.dataframe(df_p)
    
    fig2 = px.scatter(df_p, x="PC-1", y="PC-2", color=df_p.columns[-1], title='PCA biplot')
    st.plotly_chart(fig2)
    
    # model training
    st.markdown('## Model Training')
    
    if st.button('click here', help='semi_supervised learning (kmeans, catboost)'):
        with st.spinner('training models...'):
            n_clusters_hp, wcss, metrics = model_train()
            st.success('data has been preprocessed successfully')
            
        fig3 = px.line(n_clusters_hp, wcss, title='kmeans elbow plot')
        st.plotly_chart(fig3)
        
        st.write(' metrics of classification models')
        st.json(metrics)

        






