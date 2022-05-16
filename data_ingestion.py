import json
import logging
import os
import re
import shutil
import sqlite3

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
            process_type (str, optional): type of data ingestion (train, pred). Defaults to 'train'.

        Raises:
            Exception: raise exception when passing unknown process type
        """
        self.schema_dir = inges['schema']['dir']
        self.good_files_dir = inges['temp_good_dir']
        self.db_dir = inges['database']['dir']
        self.db_name = inges['database']['db_name']
        self.process_type = process_type

        # choose DSA schema file and db table name according to process type
        if self.process_type == 'train':
            self.batch_dir = inges['batch_dir_train']
            self.schema_file = inges['schema']['files']['train']
            self.db_table_name = inges['database']['tables']['train']
            self.export_folder = inges['output']['folders']['train']
            self.export_file_name = inges['output']['files']['train']
        elif self.process_type == 'pred':
            self.batch_dir = inges['batch_dir_pred']
            self.schema_file = inges['schema']['files']['pred']
            self.db_table_name = inges['database']['tables']['pred']
            self.export_folder = inges['output']['folders']['pred']
            self.export_file_name = inges['output']['files']['pred']
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

        self.good_files = []  # list of files passed the validation check
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
                                if (('int' in dtypes_cols_df[i]) or ('float' in dtypes_cols_df[i])) and (('Integer' in dtype_columns[i]) or ('float' in dtype_columns[i])):
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
                                self.good_files.append(file)

        logger.debug('data validation completed!!')

    def datatransformation(self):
        """copy good files to a temporary folder, then replace None with NULL in each file for easy data insertion

        Raises:
            Exception: raise exception if list of good files is empty
            Exception: raise exception if can not read a file via pandas
        """
        logger.debug('starting data transformation!!')
        # raise exception if no file in good files list
        if len(self.good_files) == 0:
            logger.error('no file in good_files list')
            raise Exception('no file in good_files list')

        # make temporary directory for good files
        if not os.path.exists(self.good_files_dir):  # directory check
            os.makedirs(self.good_files_dir)
        logger.debug('created temporary directory for good files')

        # copy files to the temporary directory
        for file in self.good_files:
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

    def datainsertion(self):
        """insert the content of files in the good folder to wafer.db 

        Raises:
            Exception: raise exception if can not read a file via pandas
        """
        logger.debug('starting data insertion!!')
        # make db directory
        if not os.path.exists(self.db_dir):  # directory check
            os.makedirs(self.db_dir)
        # connect to db create courser
        conn = sqlite3.connect(os.path.join(self.db_dir, self.db_name))
        cursor = conn.cursor()
        logger.debug(f'established connect with "{self.db_name}"')

        # create table
        cols_names = list(self.schema['ColName'].keys())
        cols_dtype = list(self.schema['ColName'].values())
        # change cols dtype to sqlite3 type
        cols_dtype = list(
            map(lambda x: x.replace('varchar', 'text'), cols_dtype))
        cols_dtype = list(
            map(lambda x: x.replace('float', 'real'), cols_dtype))

        cols_names_dtype = [f'"{i}" {j}' for i,
                            j in zip(cols_names, cols_dtype)]
        cols_names_dtype = ", ".join(cols_names_dtype)

        if self.process_type == 'train':
            try:
                with conn:
                    cursor.execute(
                        f"""CREATE TABLE {self.db_table_name} ({cols_names_dtype})""")
                logger.debug(f'created "{self.db_table_name}" table')
            except sqlite3.OperationalError:
                logger.warning(f'"{self.db_table_name}" table already exists')
        else:  # prediction
            try:
                with conn:
                    cursor.execute(
                        f"""CREATE TABLE {self.db_table_name} ({cols_names_dtype})""")
                logger.debug(f'created "{self.db_table_name}" table')
            except sqlite3.OperationalError:
                with conn:
                    cursor.execute(f"""DROP TABLE {self.db_table_name}""")
                    cursor.execute(
                        f"""CREATE TABLE {self.db_table_name} ({cols_names_dtype})""")
                logger.warning(
                    f'dropped "{self.db_table_name}" table and recreated it')

        # insert data to db
        for file in os.listdir(self.good_files_dir):
            # open batch file
            try:
                df = pd.read_csv(os.path.join(self.good_files_dir, file))
            except Exception:
                logger.error(f'"{file}"can not be read with pandas')
                raise Exception(f'"{file}"can not be read with pandas')
            else:
                for row in range(len(df)):
                    values_to_insert = df.iloc[row, :].to_numpy()
                    values_to_insert = [str(i) for i in values_to_insert]
                    values_ques = ', '.join(
                        ['?' for _ in range(len(values_to_insert))])
                    try:
                        with conn:
                            cursor.execute(
                                f"""INSERT INTO {self.db_table_name} VALUES ({values_ques})""", values_to_insert)
                    except Exception:
                        logger.warning(
                            f'row num {row+1} in file "{file}" can not be inserted')
                logger.debug(f'"{file}" data has been inserted to the db')

        # delete the temporary good files folder
        shutil.rmtree(self.good_files_dir, ignore_errors=True)
        logger.debug('delete good files folder')

        logger.debug('data insertion completed!!')

    def export_table_content(self):
        """export table content for either training or prediction 
        """
        # data validation
        self.data_validation()
        # data trainsformation
        self.datatransformation()
        # data insertion
        self.datainsertion()

        # get column names
        cols_names = list(self.schema['ColName'].keys())

        # connect to db create courser
        conn = sqlite3.connect(os.path.join(self.db_dir, self.db_name))
        cursor = conn.cursor()
        logger.debug(f'established connect with "{self.db_name}"')

        # query the database
        with conn:
            cursor.execute(
                f"""SELECT * from {self.db_table_name}""")
            results = cursor.fetchall()
            logger.debug('select query run successfully')

        # make output directory
        if not os.path.exists(self.export_folder):  # directory check
            os.makedirs(self.export_folder)
        logger.debug(f'created "{self.export_folder}" directory')

        # save output results to csv
        df = pd.DataFrame(results)
        df.columns = cols_names
        df.to_csv(os.path.join(self.export_folder,
                  self.export_file_name), index=False)
        logger.debug(f'"{self.export_file_name}" folder is saved successfully')

        # close db connection
        conn.close()


if __name__ == '__main__':
    data_inges = DataIngestion(process_type='pred')
    data_inges.export_table_content()
    # print(data_inges.good_files)
