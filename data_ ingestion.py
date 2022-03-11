import json
import logging
import os
import re

import pandas as pd

from logger import set_logger

# read artifacts file
with open("artifacts.json", "r") as f:
    artifacts = json.load(f)
    log = artifacts['logging']
    inges = artifacts['ingestion']

# set logger
logger = logging.getLogger(__name__)
logger = set_logger(logger, log['dir'], log['files']['data_ingestion'])


class DataIngestion:
    """data ingestion class which includes data validation (DSA), data transformation and data insertion (mysql db)
    """

    def __init__(self, process_type='train'):
        """instantiate data ingestion class

        Args:
            process_type (str, optional): type of data ingestion (training, prediction). Defaults to 'train'.

        Raises:
            Exception: raise exception when passing unknown process type
        """
        self.batch_dir = inges['batch_dir']
        self.schema_dir = inges['schema']['dir']
        self.process_type = process_type

        # choose DSA schema file according to process type
        if self.process_type == 'train':
            self.schema_file = inges['schema']['files']['train']
        elif self.process_type == 'pred':
            self.schema_file = inges['schema']['files']['pred']
        else:
            raise Exception(
                'Unknown process type. Process type should be either "train" or "pred"')

        # read schema file
        with open(os.path.join(self.schema_dir, self.schema_file), "r") as f:
            self.schema = json.load(f)

    def data_validation(self):
        # DSA checks
        # len of date and time stamp according to DSA
        len_date_stamp = self.schema['LengthOfDateStampInFile']
        len_time_stamp = self.schema['LengthOfTimeStampInFile']
        # num columns
        num_columns = self.schema['NumberofColumns']
        # column datatype
        dtype_columns = list(self.schema['ColName'].values())

        for file in os.listdir(self.batch_dir):
            # naming pattern
            pattern = '[wW]afer_\d{' + f'{len_date_stamp}' + \
                '}_\d{' + f'{len_time_stamp}' + '}.csv'
            # naming convention check
            if bool(re.search(pattern, file)) == False:
                logger.warning(
                    f'"{file}" is not matching the naming convention')
            else:
                logger.debug(
                    f'"{file}" is matching the naming convention')

                # open batch file
                try:
                    df = pd.read_csv(os.path.join(self.batch_dir, file))
                except Exception:
                    raise Exception(f'"{file}"can not be read with pandas')
                else:
                    # num columns check
                    cols = list(df.columns)
                    if len(cols) != num_columns:
                        logger.warning(
                            f'"{file}" number of columns is not matching')
                    else:
                        logger.debug(f'"{file}" number of columns is matching')
                        # ?! columns names are allowed to differ from names in DSA
                        # data type check
                        dtypes_cols_df = [str(df[col].dtypes) for col in cols]
                        type_list = []
                        for i in range(len(dtypes_cols_df)):
                            if i == 0:
                                if ('object' in dtypes_cols_df[i]) and ('varchar' in dtype_columns[i]):
                                    type_list.append('pass')
                            elif i == len(dtypes_cols_df)-1:
                                if ('int' in dtypes_cols_df[i]) and ('Integer' in dtype_columns[i]):
                                    type_list.append('pass')
                            else:
                                if (('int' in dtypes_cols_df[i]) or ('float' in dtypes_cols_df[i])) and ('float' in dtype_columns[i]):
                                    type_list.append('pass')
                        if len(type_list) != num_columns:
                            logger.warning(
                                f'"{file}" columns types are not matching')
                        else:
                            logger.debug(
                                f'"{file}" columns types are matching')
                            print(file)


if __name__ == '__main__':
    ana = DataIngestion(process_type='train')
    ana.data_validation()
