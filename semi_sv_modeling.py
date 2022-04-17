import json
import logging
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans

from logger import set_logger

# read artifacts file
with open("artifacts.json", "r") as f:
    artifacts = json.load(f)
    log = artifacts['logging']
    modeling = artifacts['modeling']
    training_dir = artifacts['ingestion']['output']['folders']['train']
    prepro_files = artifacts['preprocessing']['output_files']['train']['test']
    train_split_loc = os.path.join(training_dir, prepro_files['train_split'])
    test_split_loc = os.path.join(training_dir, prepro_files['test_split'])


# set logger
logger = logging.getLogger(__name__)
logger = set_logger(logger, log['dir'], log['files']['semi_sv_modeling'])


class SemiSV:
    """modeling class where a semi_supervised learning is used. optimized Kmeans model is used for clustering while the state of art catboost is used for classification
    """

    def __init__(self):
        """instantiating the semi_supervised learning class
        """
        self.train_loc = train_split_loc
        self.test_loc = test_split_loc
        self.modeling_dir = modeling['dir']
        self.elbow_method_name = modeling['files']['kmeans_elbow_method']
        self.clustering_model_name = modeling['files']['clustering_model']

        # make modeling dir
        if not os.path.exists(self.modeling_dir):  # directory check
            os.makedirs(self.modeling_dir)
        logger.debug('created modeling directory')

    def read_dataset(self):
        """load the training and testing dataset

        Raises:
            Exception: raise exception if the data is missing
        """
        logger.debug('starting reading the dataset!!')
        # reading the dataset
        try:
            self.train_data = pd.read_csv(self.train_loc)
            self.test_data = pd.read_csv(self.test_loc)
        except Exception:
            logger.exception(
                'cloud not manage to find the training/testing dataset')
            raise Exception(
                'cloud not manage to find the training/testing dataset')
        else:
            logger.debug('successfully loaded the dataset!!')

    def clustering(self):
        """data classification where kmeans model is used to cluster the training data after some hyperparameter tunning
        """
        logger.debug('starting clustering!!')
        # range of n_clusters
        n_clusters_hp = [i for i in range(2, 6)]
        wcss = []  # within class summation square
        # run hp optimization
        for n in n_clusters_hp:
            k_means = KMeans(n_clusters=n)
            k_means.fit(self.train_data.iloc[:, :-1])
            wcss.append(k_means.inertia_)

        # save n_clusters_hp and wcss
        elbow_data = {'n_clusters': n_clusters_hp, 'wcss': wcss}
        with open(os.path.join(self.modeling_dir, self.elbow_method_name), 'wb') as f:
            pickle.dump(elbow_data, f)
        logger.debug('saved elbow method data')

        # find knee
        self.kn = KneeLocator(n_clusters_hp, wcss,
                              curve='convex', direction='decreasing').knee
        # if no knee fount return the last item in n_clusters_hp
        if self.kn == None:
            self.kn = n_clusters_hp[-1]
            logger.debug(
                f'not found the optimum number of classes, thus used the last item in n_clusters_hp which is {self.kn}')
        else:
            logger.debug(
                f'found optimum number of classes and that is {self.kn}')

        # build the optimized kmeans model
        self.k_means = KMeans(n_clusters=self.kn)
        self.k_means.fit(self.train_data.iloc[:, :-1])
        logger.debug('build optimized kmeans model')

        # save kmeans model
        with open(os.path.join(self.modeling_dir, self.clustering_model_name), 'wb') as f:
            pickle.dump(self.k_means, f)
        logger.debug('save kmeans model')

        # predict clusters for train and test dataset
        self.train_data['cluster'] = self.k_means.predict(
            self.train_data.iloc[:, :-1])
        self.test_data['cluster'] = self.k_means.predict(
            self.test_data.iloc[:, :-1])
        logger.debug('ran the prediction for tain and test data')

        # log number of datapoints in each cluster for the train dataset
        for c in self.train_data['cluster'].unique():
            num_data_points = len(
                self.train_data[self.train_data['cluster'] == c])
            logger.debug(
                f'num data points in training dataset for cluster {c} is {num_data_points}')

        logger.debug('completed clustering!!')


if __name__ == '__main__':
    mod = SemiSV()
    mod.read_dataset()
    mod.clustering()
