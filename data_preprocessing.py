import json
import logging
import os
import pickle

import pandas as pd

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

        # make preprocessing dir
        if not os.path.exists(self.prepro_dir):  # directory check
            os.makedirs(self.prepro_dir)
        logger.debug('created for_preprocessing directory')

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
        except Exception:
            logger.exception(
                'cloud not manage to find the training/prediction data set ')
            raise Exception(
                'cloud not manage to find the training/prediction data set')
        else:
            logger.debug('successfully load the dataset!!')

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

        logger.debug('successfully completed data cleaning!!')


if __name__ == '__main__':
    data_pro = DataPreprocessor('train')
    data_pro.read_dataset()
    print(len(data_pro.data))
    data_pro.data_cleaning()
    print(data_pro.data.head())
