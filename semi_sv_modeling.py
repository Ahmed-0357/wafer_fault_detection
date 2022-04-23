import json
import logging
import os
import pickle

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from catboost.utils import eval_metric, select_threshold
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
        self.clustering_model_name = modeling['files']['clustering_model']
        self.classification_model_name = modeling['files']['classification_model']
        self.prob_cutoff_name = modeling['files']['probability_cutoff']

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
        """data clustering where kmeans model is used after some hyperparameter tunning
        """
        logger.debug('starting clustering!!')
        # range of n_clusters
        self.n_clusters_hp = [i for i in range(1, 2)]
        self.wcss = []  # within cluster summation square
        # run hp optimization
        for n in self.n_clusters_hp:
            k_means = KMeans(n_clusters=n)
            k_means.fit(self.train_data.iloc[:, :-1])
            self.wcss.append(k_means.inertia_)

        # find knee
        self.kn = KneeLocator(self.n_clusters_hp, self.wcss,
                              curve='convex', direction='decreasing').knee
        # if no knee fount return the last item in n_clusters_hp
        if self.kn == None:
            self.kn = self.n_clusters_hp[-1]
            logger.debug(
                f'not found the optimum number of clusters, thus used the last item in n_clusters_hp which is {self.kn}')
        else:
            logger.debug(
                f'found optimum number of clusters and that is {self.kn}')

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

        # log number of datapoints and classes in each cluster for the train dataset
        for c in self.train_data['cluster'].unique():
            datapoints_classes = self.train_data[self.train_data['cluster'] == c]
            n_classes = datapoints_classes.iloc[:, -2].nunique()
            logger.debug(
                f'cluster {c} in training dataset has {len(datapoints_classes)} data points with {n_classes} classes')

        logger.debug('completed clustering!!')

    def optuna_objective(self, trial, c_weights, data_train, data_test):
        """hyperparameter optimization of the main parameters in catboost classifier using optuna function

        Args:
            trial (object): optuna trial object
            c_weights (dict): classes wights
            data_train (dataframe): training data for certain cluster
            data_test (dataframe): testing data for certain cluster

        Returns:
            float: validation accuracy
        """
        learning_rate = trial.suggest_float(
            'learning_rate', 0.0001, 0.01, log=True)
        depth = trial.suggest_int('depth', 6, 10, log=False)
        l2_leaf_reg = trial.suggest_float(
            'l2_leaf_reg', 3, 12, log=False)

        # defile model
        cb_model = CatBoostClassifier(iterations=1000,      learning_rate=learning_rate,
                                      depth=depth,
                                      l2_leaf_reg=l2_leaf_reg,
                                      class_weights=c_weights, allow_writing_files=False,
                                      eval_metric='F1')

        # fit the model
        cb_model.fit(data_train.iloc[:, :-1], data_train.iloc[:, -1],
                     eval_set=(data_test.iloc[:, :-1], data_test.iloc[:, -1]), early_stopping_rounds=10, verbose=False)

        score = cb_model.best_score_
        return score['validation']['F1']

    def classification(self):
        """create classification model for each cluster after hyperparameter tunning and selection of optimum propability cut-off (low false negative rate)
        """
        logger.debug('starting classification!!')

        # dict to save prob_cutoff of each cluster
        self.prob_cutoff = {}
        # dict to save classification metrices of each cluster
        self.classification_metrices = {}
        # loop through clusters
        for c in self.train_data['cluster'].unique():
            # prep train and test data
            data_train = self.train_data[self.train_data['cluster']
                                         == c].iloc[:, :-1]
            data_test = self.test_data[self.test_data['cluster']
                                       == c].iloc[:, :-1]
            logger.debug(
                f'retrieved training and testing data for cluster {c}')

            # calculate class weights
            try:
                weight_1 = len(data_train[data_train.iloc[:, -1] == 0]) / \
                    len(data_train[data_train.iloc[:, -1] == 1])
            except Exception:  # in case class zero does not exists
                weight_1 = 1
                logger.warning(
                    'due to certain exception weight of 1 is set to 1')

            c_weights = {0: 1, 1: weight_1}
            logger.debug(f'calculated classes weights: {c_weights}')

            # hyperparameter tuning
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: self.optuna_objective(
                trial, c_weights, data_train, data_test), n_trials=5)

            best_trial = study.best_trial
            logger.debug(
                f'finished model optimization with F1 score of {best_trial.value} and the optimization parameters are {best_trial.params}')

            # build optimized model
            c_model = CatBoostClassifier(iterations=1000,      learning_rate=best_trial.params['learning_rate'],
                                         depth=best_trial.params['depth'],
                                         l2_leaf_reg=best_trial.params['l2_leaf_reg'],
                                         class_weights=c_weights, allow_writing_files=False,
                                         eval_metric='F1')

            val_pool = Pool(data_test.iloc[:, :-1], data_test.iloc[:, -1])
            c_model.fit(data_train.iloc[:, :-1], data_train.iloc[:, -1],
                        eval_set=val_pool, early_stopping_rounds=10, verbose=False)

            logging.debug(
                f"finished bulding the optimized model of cluster {c}")

            # get propability cutoff which yields false negative rate of 1%
            try:
                thre_fnr = select_threshold(c_model,
                                            data=val_pool,
                                            FNR=0.01)
            except Exception:
                thre_fnr = 0.5
                logger.warning(
                    'could not find FNR threshold, thus it was set to 0.5')

            # add propability cutoff to dict
            self.prob_cutoff[c] = thre_fnr
            logger.debug(f'FNR threshold for cluster {c} is {thre_fnr}')

            # get evaluation metrics
            train_pred = c_model.predict_proba(data_train.iloc[:, :-1])[:, 1]
            train_pred = np.where(train_pred >= thre_fnr, 1, 0)
            test_pred = c_model.predict_proba(data_test.iloc[:, :-1])[:, 1]
            test_pred = np.where(test_pred >= thre_fnr, 1, 0)

            y_train = data_train.iloc[:, -1].to_numpy()
            y_test = data_test.iloc[:, -1].to_numpy()

            prec_train = eval_metric(y_train, train_pred, 'Precision')
            prec_test = eval_metric(y_test, test_pred, 'Precision')
            recall_train = eval_metric(y_train, train_pred, 'Recall')
            recall_test = eval_metric(y_test, test_pred, 'Recall')
            self.classification_metrices[f'cluster_{c}_model'] = {'precision': {
                'train': prec_train, 'test': prec_test}, 'recall': {'train': recall_train, 'test': recall_test}}
            logger.debug(f'calculated evaluation metrics for cluster {c}')

            # save optimized model
            c_model.save_model(os.path.join(
                self.modeling_dir, f'{c}_{self.classification_model_name}'))
            logger.debug(f'saved the classification model for cluster {c}')

        # save prob_cutoff
        with open(os.path.join(self.modeling_dir, self.prob_cutoff_name), 'wb') as f:
            pickle.dump(self.prob_cutoff, f)
        logger.debug('saved prob_cutoff for all clusters')

        logger.debug("completed classification!!")

    def run(self):
        """run all the main functions in semi supervised modeling
        """
        # load preprocessed train and test dataset
        self.read_dataset()
        # perform clustering
        self.clustering()
        # perform classification
        self.classification()


if __name__ == '__main__':
    mod = SemiSV()
    mod.run()
