import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from st_app_funcs import *

# config
page_title="Wafer Fault Detection - Predict"
page_icon = ":mag_right:"
st.set_page_config(page_title = page_title,page_icon=page_icon)

# title
html_title = '<h1 align="center"> <b>📉 Data Prediction 📉</b></h1>'
st.markdown(html_title, unsafe_allow_html=True)
st.markdown('#')


# read artifacts file
with open("artifacts.json", "r") as f:
    artifacts = json.load(f)
    inges = artifacts['ingestion']
    
    schema_dir = inges['schema']['dir']
    pred_dsa_file = inges['schema']['files']['pred']
    
    pred_dir = inges["output"]['folders']['pred']
    pred_file = inges["output"]['files']['pred']
    
    pred_results_n = artifacts['prediction']["file"]
    
    mod_dir  = artifacts['modeling']['dir']
    
# data ingestion
st.markdown('## Data Ingestion')
# open training DSA
with open(os.path.join(schema_dir,pred_dsa_file)) as f:
    help_c = json.load(f)
uploaded_files = st.file_uploader('upload batch files', type='csv',
                                accept_multiple_files=True, help='the ideal file should follow this DSA format: '+str(help_c))
 
df_ip = None
if len(uploaded_files) != 0:
    with st.spinner('ingesting files...'):
        good_files = save_ingest_data(uploaded_files, type_='pred')
        st.success('files have been ingested successfully')
    st.write('files matching DSA')
    st.write(good_files)
    df_ip = pd.read_csv(os.path.join(pred_dir,pred_file))
else: # sample dataset 
    sample_data = st.checkbox(
    'use sample dataset')
    if sample_data:
        df_ip = pd.read_csv(os.path.join(pred_dir,pred_file))
        

if df_ip is not None:
    st.write('dataset')
    st.dataframe(df_ip)
    
    # data pred
    st.markdown('## Data Prediction')
    
    try:
        os.listdir(mod_dir)
    except Exception:
        st.error('no model was found, please train the model first')
    else:
        button2 = st.button('click here')
        
        if st.session_state.get('button2') != True:
            st.session_state['button2'] = button2
            
        if st.session_state['button2'] == True:
            with st.spinner('data prediction...'):
                # preprocessing
                data_process(type_='pred')
                
                # prediction
                data_predic()
                st.success('prediction been done successfully')
                
            
            # enable results downloadk
            st.markdown('###')
            st.write('prediction results')
            results_df = pd.read_csv(os.path.join(pred_dir, pred_results_n))
            st.dataframe(results_df)
            st.write('')
            st.download_button(
                label="Download prediction as CSV",
                data=results_df.to_csv(index=False).encode('utf-8'),
                file_name='prediction.csv',
                mime='text/csv',
            )
