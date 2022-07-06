import json
import os

import pandas as pd
import streamlit as st
from data_ingestion import DataIngestion

# title
html_title = '<h1 align="center"> <b>⚙️ Model Training ⚙️</b></h1>'
st.markdown(html_title, unsafe_allow_html=True)
st.markdown('#')

st.markdown('## Data')
# Load Data
# training DSA
with open('DSA_schema\schema_training.json') as f:
    help_c = json.load(f)
uploaded_files = st.file_uploader('Upload training batch files', type='csv',
                                  accept_multiple_files=True, help='the ideal file should follow this DSA format: '+str(help_c))

# save uploaded files to training_batch_files directory
if len(uploaded_files) != 0:
    with st.spinner('uploading files'):
        for uploaded_file in uploaded_files:
            df = pd.read_csv(uploaded_file)
            df.to_csv(os.path.join('training_batch_files',
                                   uploaded_file.name), index=False)
        st.success('files has been uploaded successfully')
    
else:
    sample_data = st.checkbox(
        'use sample dataset')
    if sample_data:
        df = pd.read_csv(r'training_data\train_raw.csv')
        st.dataframe(df)

if len(uploaded_files) > 0:
    # data ingestion
    st.markdown('#')
    st.markdown('## Data Ingestion')
    if st.button('Click To Ingest the Data', help='data ingestion includes data validation with DSA, data transformation and data insertion into mysql db'):
        with st.spinner('running data ingestion'):
            data_inges = DataIngestion(process_type='train')
            data_inges.export_table_content()
            st.write(data_inges.good_files)
