import json
import logging
import os
import pickle

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from logger import set_logger

# read artifacts file
with open("artifacts.json", "r") as f:
    artifacts = json.load(f)
    log = artifacts['logging']
    data_prepro = artifacts['preprocessing']
    # data location
    output_artifacts = artifacts['ingestion']['output']
    train_location = os.path.join(
        output_artifacts['folders']['train'], output_artifacts['files']['train'])
    pred_location = os.path.join(
        output_artifacts['folders']['pred'], output_artifacts['files']['pred'])


# set logger
logger = logging.getLogger(__name__)
logger = set_logger(logger, log['dir'], log['files']['data_preprocessing'])


class DataPreprocessor:
    """data preprocessing includes data cleaning, train_test split, missing data imputation and features scaling
    """

    def __init__(self, process_type='train'):
        """instantiaion of data preprocessing class

        Args:
            process_type (str, optional): process type can be either train or pred. Defaults to 'train'.
        """
        self.process_type = process_type
        self.prepro_dir = data_prepro['dir']
        self.zero_std_cols = data_prepro['files']['removed_cols']
        self.train_data_dir = output_artifacts['folders']['train']
        self.pred_data_dir = output_artifacts['folders']['pred']
        self.train_raw_name = data_prepro['output_files']['train']['train_raw']
        self.pred_raw_name = data_prepro['output_files']['pred']['pred_raw']
        self.imputer_name = data_prepro['files']['data_imputer']
        self.scaler_name = data_prepro['files']['data_scaler']

        self.train_split_name = data_prepro['output_files']['train']['test']['train_split']
        self.test_split_name = data_prepro['output_files']['train']['test']['test_split']
        self.pred_prepro_name = data_prepro['output_files']['pred']['pred_prepro']

        # make preprocessing dir
        if not os.path.exists(self.prepro_dir):  # directory check
            os.makedirs(self.prepro_dir)
        logger.debug('created preprocessing directory')

    def read_dataset(self):
        """load the training/prediction dataset

        Raises:
            Exception: raise exception in case of unknown process type
            Exception: raise exception if the data is missing
        """
        logger.debug('starting reading the dataset!!')
        # get the correct path
        if self.process_type == 'train':
            data_path = train_location
        elif self.process_type == 'pred':
            data_path = pred_location
        else:
            logger.error(
                'invalid process_type, process type should be either train or pred')
            raise Exception(
                'invalid process_type, process type should be either train or pred')
        # reading the dataset
        try:
            self.data = pd.read_csv(data_path)
            # this will be showed for the user at the end with prediction
            self.data_raw = self.data.copy()
        except Exception:
            logger.exception(
                'cloud not manage to find the training/prediction dataset')
            raise Exception(
                'cloud not manage to find the training/prediction dataset')
        else:
            logger.debug('successfully loaded the dataset!!')

    def data_cleaning(self):
        """training dataset cleaning includes removing wafer column, removing duplicates rows, fix output column labeling, removing columns with single values.
        prediction dataset cleaning includes removing wafer column, removing duplicates rows, fix output column labeling
        """
        logger.debug('starting dataset cleaning!!')
        # remove wafter column
        del self.data[self.data.columns[0]]
        logger.debug('deleted Wafer column')

        # remove duplicates rows
        self.data.drop_duplicates(inplace=True)
        logger.debug('removed duplicates rows')

        # fix output column labeling (0:good wafer, 1:bad wafer)
        self.data[self.data.columns[-1]
                  ] = self.data[self.data.columns[-1]].replace([-1], 0)
        logger.debug('replaced -1 with 0 in output column')

        # remove columns with single value
        if self.process_type == 'train':
            # calculate std value for the features
            cols_std = self.data.iloc[:, :-1].std().to_dict()
            cols_removed = []
            for col, std_v in zip(cols_std.keys(), cols_std.values()):
                if std_v == 0:
                    cols_removed.append(col)
                    del self.data[col]
                    logger.debug(
                        f'removed "{col}" column from the train data because it has zero std')
            logger.debug(
                f'total of {len(cols_removed)} columns has been removed from training data')
            # save zero std cols
            with open(os.path.join(self.prepro_dir, self.zero_std_cols), 'wb') as f:
                pickle.dump(cols_removed, f)
                logger.debug('saved removed columns')
        else:
            # load zero std cols
            with open(os.path.join(self.prepro_dir, self.zero_std_cols), 'rb') as f:
                cols_removed = pickle.load(f)
                for col in cols_removed:
                    del self.data[col]
                    logger.debug(
                        f'removed "{col}" column from the prediction data because it has zero std')
                logger.debug(
                    f'total of {len(cols_removed)} columns has been removed from prediction data')

        # save the row data
        new_index = self.data.index.tolist()
        self.data_raw = self.data_raw.iloc[new_index, :]
        if self.process_type == 'train':  # use the relabeled output
            self.data_raw.iloc[:, -1] = self.data.iloc[:, -1]
            self.data_raw.to_csv(os.path.join(
                self.train_data_dir, self.train_raw_name), index=False)
            logger.debug("saved raw train data")
        else:
            self.data_raw.to_csv(os.path.join(
                self.pred_data_dir, self.pred_raw_name), index=False)
            logger.debug("saved raw pred data")

        logger.debug('successfully completed data cleaning!!')

    def data_split(self):
        """train test split of the training data (70 to 30)
        """
        logger.debug("starting data splitting!!")
        if self.process_type == 'train':
            col_names = self.data.columns
            X_train, X_test, y_train, y_test = train_test_split(
                self.data.iloc[:, :-1], self.data.iloc[:, -1], test_size=0.3, stratify=self.data.iloc[:, -1])

            # save train split data
            self.train_data = pd.DataFrame(X_train)
            self.train_data[col_names[-1]] = y_train
            self.train_data.columns = col_names

            # save test split data
            self.test_data = pd.DataFrame(X_test)
            self.test_data[col_names[-1]] = y_test
            self.test_data.columns = col_names
            logger.debug(
                f"length of training data: {len(self.train_data)} - length of testing data: {len(self.test_data)}")
        else:
            self.pred_data = self.data.copy()
            logger.debug(f"length of prediction data: {len(self.pred_data)}")

        logger.debug("completed data splitting!!")

    def missing_data_imp(self):
        """knn imputer is used to filling missing features data
        """
        logger.debug("starting missing data imputation!!")

        if self.process_type == 'train':
            # fit imputer
            imputer = KNNImputer(n_neighbors=5, weights='distance')
            imputer.fit(self.train_data.iloc[:, :-1])
            logger.debug('trained knn imputer')
            # fill missing in train and test data
            self.train_data[self.train_data.columns[:-1]
                            ] = imputer.transform(self.train_data.iloc[:, :-1])
            self.test_data[self.test_data.columns[:-1]
                           ] = imputer.transform(self.test_data.iloc[:, :-1])
            logger.debug('filled missing values in train and test data')
            # save imputer
            with open(os.path.join(self.prepro_dir, self.imputer_name), 'wb') as f:
                pickle.dump(imputer, f)
            logger.debug('saved knn imputer')
        else:
            # load imputer
            with open(os.path.join(self.prepro_dir, self.imputer_name), 'rb') as f:
                imputer = pickle.load(f)
            logger.debug('loaded knn imputer')
            self.pred_data[self.pred_data.columns
                           ] = imputer.transform(self.pred_data)
            logger.debug('filled missing values in prediction data')

        logger.debug("completed filling missing data!!")

    def features_scaling(self):
        """min max normalization of the features
        """
        logger.debug("starting features scaling!!")
        if self.process_type == 'train':
            # fit scaler
            scaler = MinMaxScaler()
            scaler.fit(self.train_data.iloc[:, :-1])
            logger.debug("fitted the scaler")
            # scale train and test data
            self.train_data[self.train_data.columns[:-1]
                            ] = scaler.transform(self.train_data.iloc[:, :-1])
            self.test_data[self.test_data.columns[:-1]
                           ] = scaler.transform(self.test_data.iloc[:, :-1])
            logger.debug('scaled the values in train and test data')
            # save the scaler
            with open(os.path.join(self.prepro_dir, self.scaler_name), 'wb') as f:
                pickle.dump(scaler, f)
            logger.debug('saved the scaler')
        else:
            # load scaler
            with open(os.path.join(self.prepro_dir, self.scaler_name), 'rb') as f:
                scaler = pickle.load(f)
            logger.debug('loaded the scaler')
            self.pred_data[self.pred_data.columns
                           ] = scaler.transform(self.pred_data)
            logger.debug('scaled the values in prediction data')

        logger.debug("completed features scaling!!")

    def run(self):
        """run all the main functions for data preprocessing and save the data for the next module
        """
        # read the dataset
        self.read_dataset()
        # data cleaning
        self.data_cleaning()
        # data split
        self.data_split()
        # fill missing
        self.missing_data_imp()
        # features scaling
        self.features_scaling()
        # save the files
        if self.process_type == 'train':
            self.train_data.to_csv(os.path.join(
                self.train_data_dir, self.train_split_name), index=False)
            logging.debug("saved training data")

            self.test_data.to_csv(os.path.join(self.train_data_dir,
                                               self.test_split_name), index=False)
            logging.debug("saved test data")
        else:
            self.pred_data.to_csv(os.path.join(
                self.pred_data_dir, self.pred_prepro_name), index=False)
            logging.debug("saved prediction data")


if __name__ == '__main__':
    data_pro = DataPreprocessor('train')
    data_pro.run()
