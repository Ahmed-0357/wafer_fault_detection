import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from st_app_funcs import *

# config
page_title="Wafer Fault Detection - Train"
page_icon = ":mag_right:"
st.set_page_config(page_title = page_title,page_icon=page_icon)

# title
html_title = '<h1 align="center"> <b>⚙️ Model Training ⚙️</b></h1>'
st.markdown(html_title, unsafe_allow_html=True)
st.markdown('#')

# read artifacts file
with open("artifacts.json", "r") as f:
    artifacts = json.load(f)
    inges = artifacts['ingestion']
    
    schema_dir = inges['schema']['dir']
    train_dsa_file = inges['schema']['files']['train']
    
    train_dir = inges["output"]['folders']['train']
    train_file = inges["output"]['files']['train']
    
    train_split_file = artifacts['preprocessing']['output_files']['train']['splits']['train_split']

# data ingestion
st.markdown('## Data Ingestion')
# open training DSA
with open(os.path.join(schema_dir,train_dsa_file)) as f:
    help_c = json.load(f)
uploaded_files = st.file_uploader('upload batch files', type='csv',
                                accept_multiple_files=True, help='the ideal file should follow this DSA format: '+str(help_c))
 
df_i, df_p = None, None
if len(uploaded_files) != 0:
    with st.spinner('ingesting files...'):
        good_files = save_ingest_data(uploaded_files, type_='train')
        st.success('files have been ingested successfully')
    st.write('files matching DSA')
    st.write(good_files)
    df_i = pd.read_csv(os.path.join(train_dir,train_file))
else: # sample dataset 
    sample_data = st.checkbox(
    'use sample dataset')
    if sample_data:
        df_i = pd.read_csv(os.path.join(train_dir,train_file))

if df_i is not None:
    df_i.iloc[:,-1] = df_i.iloc[:,-1].replace([-1], 0)
    df_i.iloc[:,-1] = df_i.iloc[:,-1].astype('str')
    st.write('dataset')
    st.dataframe(df_i)
    
    # percentages of output values 
    fig1 = px.pie(df_i, names=df_i.columns[-1], title='output column')
    st.plotly_chart(fig1)
            

    # data preprocessing
    st.markdown('## Data Preprocessing')
    button1 = st.button('click here', help='data preprocessing includes data cleaning, train-validation-test split, missing data imputation, features scaling and PCA decomposition')
    
    if st.session_state.get('button') != True:
        st.session_state['button'] = button1
    
    if st.session_state['button'] == True:
        with st.spinner('data preprocessing...'):
            data_process(type_='train')
            st.success('data has been preprocessed successfully')
            df_p = pd.read_csv(os.path.join(train_dir,train_split_file))
            
        if df_p is not None:
            st.write('preprocessed training split')
            df_p.iloc[:,-1] = df_p.iloc[:,-1].astype('str')
            st.dataframe(df_p)
            
            # PCA biplot 
            fig2 = px.scatter(df_p, x="PC-1", y="PC-2", color=df_p.columns[-1], title='PCA biplot')
            st.plotly_chart(fig2)
            
            # model training
            st.markdown('## Model Training')
            
            if st.button('click here', help='semi_supervised learning (clustering and classification)'):
                with st.spinner('training models...'):
                    n_clusters_hp, wcss, df_tc, metrics = model_train()
                    st.success('models have been trained successfully')
                
                # kmeans elbow plot
                fig3 = px.line(x=n_clusters_hp, y=wcss, labels={
                     "x": "number of clusters",
                     "y": "WCSS"},title='kmeans elbow plot')
                st.plotly_chart(fig3)
                
                # clusters plot
                df_tc['cluster'] = df_tc['cluster'].astype('str')
                fig4 = px.scatter_3d(df_tc, x='PC-1', y='PC-2', z='PC-3', color='cluster', title='clusters distribution')
                st.plotly_chart(fig4)
                
                # metrics output
                st.write('metrics of classification models')
                st.json(metrics)
                st.session_state['button'] = True







