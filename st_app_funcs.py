import os

import pandas as pd
import streamlit as st

from data_ingestion import DataIngestion
from data_preprocessing import DataPreprocessor
from semi_sv_modeling import SemiSV


@st.cache(show_spinner=False)
def save_ingest_data(uploaded_files):
    """saving and ingesting training batch files

    Args:
        uploaded_files (list): list of uploaded files

    Returns:
        list: list of good files matching DSA 
    """
    # save uploaded files to training_batch_files directory
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        df.to_csv(os.path.join('training_batch_files',
                               uploaded_file.name), index=False)
    # data ingestion
    data_inges = DataIngestion(process_type='train')
    data_inges.export_table_content()
    good_files = data_inges.good_files

    return good_files


@st.cache(show_spinner=False)
def data_process():
    """preprocessing the data
    """
    data_pro = DataPreprocessor('train')
    data_pro.run()


@st.cache(show_spinner=False)
def model_train():
    """model training

    Returns:
        list, dict : list contains differ number of clusters and the corrsponding wcss, evaluation metrics
    """
    mod = SemiSV()
    mod.run()
    n_clusters_hp = mod.n_clusters_hp
    wcss = mod.wcss
    metrics = mod.classification_metrices

    return [n_clusters_hp, wcss], metrics
