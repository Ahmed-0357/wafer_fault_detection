import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from logger import set_logger

# read artifacts file
with open("artifacts.json", "r") as f:
    artifacts = json.load(f)
    log = artifacts['logging']
    modeling = artifacts['modeling']
    pred_dir = artifacts['ingestion']['output']['folders']['pred']
    prepro_file = artifacts['preprocessing']['output_files']['pred']['pred_prepro']
    pred_raw_file = artifacts['preprocessing']['output_files']['pred']['pred_raw']
    pred_loc = os.path.join(pred_dir, prepro_file)
    pred_raw_loc = os.path.join(pred_dir, pred_raw_file)


# set logger
logger = logging.getLogger(__name__)
logger = set_logger(logger, log['dir'], log['files']['prediction'])


class Prediction:
    """this class for new data prediction, the prediction is semi supervised where both clustering and classification models will be used
    """

    def __init__(self):
        """instantiating the prediction class
        """
        self.pred_loc = pred_loc

        self.modeling_dir = modeling['dir']
        self.clustering_model_name = modeling['files']['clustering_model']
        self.classification_model_name = modeling['files']['classification_model']
        self.prob_cutoff_name = modeling['files']['probability_cutoff']

        self.pred_raw_loc = pred_raw_loc
        self.pred_dir = pred_dir
        self.pred_results_name = artifacts['prediction']['file']

    def read_dataset(self):
        """load prediction dataset

        Raises:
            Exception: raise exception if the data is missing
        """
        logger.debug('starting reading the dataset!!')
        # reading the dataset
        try:
            self.pred_data = pd.read_csv(self.pred_loc)
            self.pred_raw_df = pd.read_csv(self.pred_raw_loc)
        except Exception:
            logger.exception(
                'cloud not manage to find the prediction dataset')
            raise Exception(
                'cloud not manage to find the prediction dataset')
        else:
            logger.debug('successfully loaded the dataset!!')

    def clustering(self):
        """predict the cluster of each data point
        """
        logger.debug('starting clustering!!')

        # load kmeans model
        with open(os.path.join(self.modeling_dir, self.clustering_model_name), 'rb') as f:
            k_means = pickle.load(f)
        logger.debug('save kmeans model')

        # predict clusters
        self.pred_data['cluster'] = k_means.predict(
            self.pred_data)

        logger.debug('ran the clusters prediction')
        logger.debug('completed clustering!!')

    def classification(self):
        """pred the status of each wafer
        """
        logger.debug('starting classification!!')

        # load probability_cutoff
        with open(os.path.join(self.modeling_dir, self.prob_cutoff_name), 'rb') as f:
            prob_cutt_off = pickle.load(f)
        logger.debug('loaded prob_cutoff')

        # list contains all dfs of indexes and prediction
        all_pred_dfs = []
        # loop through all clusters
        for c in self.pred_data['cluster'].unique():
            data_pred = self.pred_data[self.pred_data['cluster']
                                       == c].iloc[:, :-1].copy()

            # load catboost model
            cb = CatBoostClassifier()
            cb.load_model(os.path.join(self.modeling_dir,
                          f'{c}_{self.classification_model_name}'))
            logger.debug(f'loaded catbooost model for cluster {c}')

            # run prediction
            pred_v = cb.predict_proba(data_pred)[:, 1]
            pred_v = np.where(pred_v >= prob_cutt_off[c], 1, 0)
            data_pred['prediction'] = pred_v
            data_pred.reset_index(drop=False, inplace=True)
            all_pred_dfs.append(data_pred[['index', 'prediction']])
            logger.debug(f'ran prediction for cluster {c}')

        # merge prediction of all clusters
        for i in range(len(all_pred_dfs)):
            if i == 0:
                self.df = all_pred_dfs[0]
            else:
                df_ = all_pred_dfs[i]
                self.df = pd.concat([self.df, df_], ignore_index=True)

        self.df.sort_values('index', inplace=True)

        logger.debug('merged prediction of all clusters')
        logger.debug("completed classification!!")

    def run(self):
        """run all the main functions prediction class
        """
        # load preprocessed train and test dataset
        self.read_dataset()
        # perform clustering
        self.clustering()
        # perform classification
        self.classification()
        # save results
        self.pred_raw_df['prediction'] = self.df['prediction'].to_numpy()
        self.pred_raw_df.to_csv(os.path.join(
            self.pred_dir, self.pred_results_name), index=False)


if __name__ == '__main__':
    pred_c = Prediction()
    pred_c.run()
