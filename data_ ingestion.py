import json
import logging
import os
import re
import shutil

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
        self.good_files_dir = inges['temp_good_dir']

        # choose DSA schema file according to process type
        if self.process_type == 'train':
            self.schema_file = inges['schema']['files']['train']
        elif self.process_type == 'pred':
            self.schema_file = inges['schema']['files']['pred']
        else:
            logger.error(
                'Unknown process type. Process type should be either "train" or "pred"')
            raise Exception(
                'Unknown process type. Process type should be either "train" or "pred"')

        # read schema file
        with open(os.path.join(self.schema_dir, self.schema_file), "r") as f:
            self.schema = json.load(f)

    def data_validation(self):
        """validate if batch files match with client DSA agreement - naming convention, number of columns, columns datatype, full empty columns

        Raises:
            Exception: raise exception batch files directory is empty
            Exception: raise exception if can not read a file via pandas

        Returns:
            list: list of good files which have passed all validation checks
        """
        logger.debug('staring data validation!!')
        # len of date and time stamp according to DSA
        len_date_stamp = self.schema['LengthOfDateStampInFile']
        len_time_stamp = self.schema['LengthOfTimeStampInFile']
        # num columns
        num_columns = self.schema['NumberofColumns']
        # column datatype
        dtype_columns = list(self.schema['ColName'].values())

        # raise error if batch dir is empty
        list_files = os.listdir(self.batch_dir)
        if len(list_files) == 0:
            logger.error(f'"{self.batch_dir}" directory has no files')
            raise Exception(f'"{self.batch_dir}" directory has no files')

        validation_pass = []  # list of files passed the validation check
        for file in list_files:
            # naming convention check
            # naming pattern
            pattern = '[wW]afer_\d{' + f'{len_date_stamp}' + \
                '}_\d{' + f'{len_time_stamp}' + '}.csv'
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
                    logger.error(f'"{file}"can not be read with pandas')
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
                        type_pass = []
                        for i in range(len(dtypes_cols_df)):
                            if i == 0:
                                if ('object' in dtypes_cols_df[i]) and ('varchar' in dtype_columns[i]):
                                    type_pass.append('pass')
                            elif i == (len(dtypes_cols_df)-1):
                                if ('int' in dtypes_cols_df[i]) and ('Integer' in dtype_columns[i]):
                                    type_pass.append('pass')
                            else:
                                if (('int' in dtypes_cols_df[i]) or ('float' in dtypes_cols_df[i])) and ('float' in dtype_columns[i]):
                                    type_pass.append('pass')
                        if len(type_pass) != num_columns:
                            logger.warning(
                                f'"{file}" columns types are not matching')
                        else:
                            logger.debug(
                                f'"{file}" columns types are matching')
                            # empty columns check
                            max_null_count = df.isnull().sum().to_numpy().max()
                            num_rows = len(df)
                            if num_rows == max_null_count:
                                logger.warning(
                                    f'"{file}" has full empty column(s)')
                            else:
                                logger.debug(
                                    f'"{file}" passed empty columns check')
                                validation_pass.append(file)

        logger.debug('data validation completed!!')
        return validation_pass

    def DataTransformation(self, good_files):
        """copy list of good files to a temporary folder, then replace None with NULL in each file for easy data insertion

        Args:
            good_files (list): list of good files that passed data validation checks

        Raises:
            Exception: raise exception if list of good files is empty
            Exception: raise exception if can not read a file via pandas
        """
        logger.debug('starting data transformation!!')
        # raise exception if no file in good files list
        if len(good_files) == 0:
            logger.error('no file in good_files list')
            raise Exception('no file in good_files list')

        # make temporary directory for good files
        if not os.path.exists(self.good_files_dir):  # directory check
            os.makedirs(self.good_files_dir)
        logger.debug('created temporary directory for good files')

        # copy files to the temporary directory
        for file in good_files:
            original_loc = os.path.join(self.batch_dir, file)
            target_loc = os.path.join(self.good_files_dir, file)
            shutil.copyfile(original_loc, target_loc)
            logger.debug(f'copied "{file}" file to good files folder')

        # replace None values with Null for easy data insertion
        for file in os.listdir(self.good_files_dir):
            # open batch file
            try:
                df = pd.read_csv(os.path.join(self.good_files_dir, file))
            except Exception:
                logger.error(f'"{file}"can not be read with pandas')
                raise Exception(f'"{file}"can not be read with pandas')
            else:
                df.fillna('NULL', inplace=True)
                df.to_csv(os.path.join(self.good_files_dir, file), index=False)
                logger.debug(f'replaced none with null in file "{file}"')

        logger.debug('data transformation completed!!')

    def DataInsertion(self):
        pass


if __name__ == '__main__':
    data_inges = DataIngestion(process_type='train')
    good_files = data_inges.data_validation()
    print(f'\nGood Files: {good_files}')
    data_inges.DataTransformation(good_files)
