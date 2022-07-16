import json
import os
import shutil

import pandas as pd
import streamlit as st

from data_ingestion import DataIngestion
from data_preprocessing import DataPreprocessor
from prediction import Prediction
from semi_sv_modeling import SemiSV

# read artifacts file
with open("artifacts.json", "r") as f:
    artifacts = json.load(f)
    inges = artifacts['ingestion']

    batch_dir_train = inges['batch_dir_train']
    batch_dir_pred = inges['batch_dir_pred']


@st.cache(show_spinner=False)
def save_ingest_data(uploaded_files, type_='train'):
    """saving and ingesting training batch files

    Args:
        uploaded_files (list): list of uploaded files
        type_ (str, optional): process type, and it can be either train or pred. Defaults to 'train'.

    Returns:
        list: list of good files matching DSA
    """
    if type_ == 'train':
        dir_ = batch_dir_train
    else:
        dir_ = batch_dir_pred

    # delete dir_ if it does exist or create a new one
    try:
        shutil.rmtree(dir_)
        os.makedirs(dir_)
    except OSError:
        os.makedirs(dir_)
    # save uploaded files to training_batch_files directory
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        df.to_csv(os.path.join(dir_,
                               uploaded_file.name), index=False)
    # data ingestion
    if type_ == 'train':
        data_inges = DataIngestion(process_type='train')
    else:
        data_inges = DataIngestion(process_type='pred')

    data_inges.export_table_content()
    good_files = data_inges.good_files

    return good_files


@st.cache(show_spinner=False)
def data_process(type_='train'):
    """preprocessing the data

    Args:
        type_ (str, optional): process type, and it can be either train or pred. Defaults to 'train'.
    """
    if type_ == 'train':
        data_pro = DataPreprocessor('train')
    else:
        data_pro = DataPreprocessor('pred')

    data_pro.run()


st.cache(show_spinner=False)


def model_train():
    """model training

    Returns:
        list, list, Dataframe, dict : list contains differ number of clusters, list contains the corresponding wcss, train dataframe with clustering results,  evaluation metrics
    """
    mod = SemiSV()
    mod.run()
    n_clusters_hp = mod.n_clusters_hp
    wcss = mod.wcss
    metrics = mod.classification_metrices
    df_tc = mod.train_data

    return n_clusters_hp, wcss, df_tc, metrics


@st.cache(show_spinner=False)
def data_predic():
    """run prediction for blind data
    """
    pred_c = Prediction()
    pred_c.run()
