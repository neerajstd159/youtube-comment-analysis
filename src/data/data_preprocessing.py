import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
import os
nltk.download('wordnet')
nltk.download('stopwords')
import logging

logger = logging.Logger('data_preprocessing')
logger.setLevel(logging.DEBUG)

# file handler
fileHandler = logging.FileHandler('errors.log')
fileHandler.setLevel(logging.ERROR)

# console handler
streamHandler = logging.StreamHandler()
streamHandler.setLevel(logging.DEBUG)

# formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
streamHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)

# add handler to logger
logger.addHandler(streamHandler)
logger.addHandler(fileHandler)


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


def preprocess_df(comment: str) -> str:
    """Apply preprocessing transformations in a comment"""
    try:
        # convert to lowercase
        comment = comment.lower()

        # remove leading and trailing whitespace
        comment = comment.strip()

        # remove newline chars
        comment = re.sub(r'\n', ' ', comment)

        # remove non alpha-numeric chars except punctuation
        comment = re.sub(r'[^a-zA-Z0-9\s,.!?]', '', comment)

        # remove some stopwords
        stopwords = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stopwords])

        # apply lemmatization
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment    
    except Exception as e:
        logger.error('Error in preprocessing comment %s', e)
        return comment


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing transformation in dataset"""
    try:
        df['clean_comment'] = df['clean_comment'].apply(preprocess_df)
        logger.debug("Text normalization completed")

        return df
    except Exception as e:
        logger.error("Error during data preprocessing: %s", e)
        raise


def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, path: str) -> None:
    """Save the train and test datasets inside interim"""
    try:
        # create interim dir if it doesn't exist
        interim_data_path = os.path.join(path, 'interim')
        os.makedirs(interim_data_path, exist_ok=True)

        # save datsets
        train_df.to_csv(os.path.join(interim_data_path, 'train_processed.csv'), index=False)
        test_df.to_csv(os.path.join(interim_data_path, 'test_processed.csv'), index=False)

        logger.debug('train and test datasets saved to interim dir successfully')
    except Exception as e:
        logger.error('Unexpected error while saving data to interim: %s', e)
        raise


def main():
    try:
        # fetch train and test datasets from raw dir
        train_df = load_data('./data/raw/train.csv')
        test_df = load_data('./data/raw/test.csv')

        # apply preprocessing
        train_processed_df = normalize_text(train_df)
        test_processed_df = normalize_text(test_df)

        # save datasets
        save_data(train_processed_df, test_processed_df, './data')
    except Exception as e:
        logger.error('Failed to complete preprocessing stage: %s', e)


if __name__ == '__main__':
    main()