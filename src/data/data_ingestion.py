import numpy as np
import pandas as pd
import os
import yaml
import logging
from sklearn.model_selection import train_test_split

# logging
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

# stream handler
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG)

# console handler
streamHandler = logging.FileHandler('errors.log')
streamHandler.setLevel(logging.ERROR)

# formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consoleHandler.setFormatter(formatter)
streamHandler.setFormatter(formatter)

# add handler to logger
logger.addHandler(consoleHandler)
logger.addHandler(streamHandler)

def load_params(params_path: str) -> dict:
    """load parameters value from given yaml file"""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters retrieved from %s", params_path)
        return params
    except FileNotFoundError:
        logger.error("File not found for given path: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("Yaml error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise


def load_data(data_path: str) -> pd.DataFrame:
    """Load dataset from given CSV File"""
    try:
        df = pd.read_csv(data_path)
        logger.debug("Dataset successfully loaded")
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file: %s", e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handling missing values, duplicates and empty string"""
    try:
        # missing values
        df = df.dropna()
        # drop duplicates
        df = df.drop_duplicates()
        # empty string
        df = df[df['clean_comment'].str.strip() != '']

        logger.debug("Basic preprocessing completed")
        return df
    except KeyError as e:
        logger.error('Missing column in dataset %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, path: str) -> None:
    """Save the train and test dataset inside raw folder"""
    try:
        # create raw dir if it does not exist
        raw_data_path = os.path.join(path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)

        # save the dataset
        train_df.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)

        logger.debug('Train and Test dataset saved successfully inside raw dir')
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def main():
    try:
        # load params from params.yaml file in the root dir
        params = load_params(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../params.yaml')))
        test_size = params['data_ingestion']['test_size']

        # Load dataset
        df = load_data('https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')

        # Basic data cleaning
        processed_df = preprocess_data(df)

        # Split the data
        train_df, test_df = train_test_split(processed_df, test_size=test_size, random_state=42)

        # Save training and testing datasets
        save_data(train_df, test_df, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data')))

    except Exception as e:
        logger.error('Failed to complete data ingestion stage: %s', e)


if __name__ == '__main__':
    main()