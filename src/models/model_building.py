import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
import os
import yaml
import logging
import pickle

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


def load_params(params_path: str) -> dict:
    """Load parameters from given yaml file"""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('parameters loaded successfully')

        return params
    except FileNotFoundError:
        logger.error('File not Found for given url: %s', params_path)
        raise


def load_data(path: str) -> pd.DataFrame:
    """Load dataset from given CSV File"""
    try:
        df = pd.read_csv(path)
        logger.debug("Dataset successfully loaded")
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file: %s", e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def apply_tfidf(train_df: pd.DataFrame, max_feature: int, ngram_rang: tuple) -> tuple:
    """Apply tfidf vectorizer to training data"""
    try:
        # split data
        X_train = train_df['clean_comment'].fillna("").astype(str).values
        y_train = train_df['category'].values

        # vectorizer
        vectorizer = TfidfVectorizer(max_features=max_feature, ngram_range=ngram_rang)
        X_train_trf = vectorizer.fit_transform(X_train)
        logger.debug('vectorizer transformaion to train data completed')

        # save vectorizer
        root_path = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))
        with open(os.path.join(root_path, 'tfidf_vectorizer.pkl'), 'wb') as file:
            pickle.dump(vectorizer, file)
        logger.debug('vectorizer saved to root dir')

        return X_train_trf, y_train
    except Exception as e:
        logger.error('Error while vectorization: %s', e)
        raise


def train_model(X_train: np.ndarray, y_train: np.ndarray, lr: float, max_depth: int, n_estimators: int) -> LGBMClassifier:
    """Train the lightgbm model"""
    try:
        model = LGBMClassifier(
            objective='multiclass',
            num_class=3,
            metric='multi_logloss',
            is_unbalance=True,
            class_weight='balanced',
            reg_alpha=0.1,
            reg_lambda=0.1,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=lr
        )
        model.fit(X_train, y_train)
        logger.debug('model training completed')

        return model
    except Exception as e:
        logger.error('Error while training the model: %s', e)
        raise


def save_model(model, path: str) -> None:
    """Save the model"""
    try:
        with open(path, 'wb') as file:
            pickle.dump(model, file)

        logger.debug('model saved to %s', e)
    except Exception as e:
        logger.error('Error occured while saving the model: %s', e)
        raise


def main():
    try:
        root_dir = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))

        # load params
        params = load_params(os.path.join(root_dir, 'params.yaml'))

        max_features = params['model_building']['max_features']
        ngram_range = tuple(params['model_building']['ngram_range'])
        learning_rate = params['model_building']['learning_rate']
        max_depth = params['model_building']['max_depth']
        n_estimators = params['model_building']['n_estimators']

        # load dataset
        train_data = load_data(os.path.join(root_dir, 'data/interim/train_processed.csv'))

        # apply vectorizer
        X_train_trf, y_train = apply_tfidf(train_data, max_features, ngram_range)

        # train model
        model = train_model(X_train_trf, y_train, learning_rate, max_depth, n_estimators)

        # save model
        save_model(model, os.path.join(root_dir, 'lgbm_model.pkl'))
    except Exception as e:
        logger.error('Unexpected error occured in model building stage: %s', e)


if __name__ == '__main__':
    main()