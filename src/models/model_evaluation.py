import numpy as np
import pandas as pd
import pickle
import json
import os
import logging
import mlflow
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

streamHandler = logging.StreamHandler()
streamHandler.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler('errors.log')
fileHandler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
streamHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)

logger.addHandler(streamHandler)
logger.addHandler(fileHandler)


def load_params(path: str) -> dict:
    """Load params from given yaml File"""
    try:
        with open(path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('params loaded successfully')

        return params
    except Exception as e:
        logger.error('Unexpected error while loading params: %s', e)
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


def load_model(path: str):
    """Load the model"""
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('model loaded successfully')

        return model
    except Exception as e:
        logger.error('Error while loading model: %s', e)
        raise


def load_vectorizer(path: str) -> TfidfVectorizer:
    """Load the tfidf vectorizer"""
    try:
        with open(path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug('vectorizer loaded successfully')

        return vectorizer
    except Exception as e:
        logger.error('Error while loading vectorizer: %s', e)
        raise


def evaluate_model(model, X_test: np.ndarray, y_test:np.ndarray):
    """Evaluate the model and returns classification report and confusion matrics"""
    try:
        y_pred = model.predict(X_test)
        class_report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        logger.debug('model evaluation completed')
        return class_report, conf_matrix
    except Exception as e:
        logger.error('Error in model evaluation: %e', e)
        raise


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run_id and path to json file"""
    try:
        model_info = {
            "run_id":run_id,
            "model_path":model_path
        }
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error while saving model info')
        raise


def main():
    # mlflow setup
    mlflow.set_registry_uri('')
    mlflow.set_experiment('')

    with mlflow.start_run() as run:
        try:
            # load params
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
            params = load_params(os.path.join(root_dir, 'params.yaml'))
            
            # log params
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            # load model
            model = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))

            # load vectorizer
            vectorizer = load_vectorizer(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # load and transform test data
            test_df = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))
            X_test = test_df['clean_comment'].values
            y_test = test_df['category'].values
            X_test_trf = vectorizer.transform(X_test)

            # input dataframe for signature and log model
            input_df = pd.DataFrame(X_test_trf.toarray()[:5], columns=vectorizer.get_feature_names_out())
            signature = mlflow.models.infer_signature(input_df, model.predict(X_test_trf[:5]))

            # log model
            mlflow.sklearn.log_model(model, 'lgbm_model', signature=signature, input_example=input_df)

            # save model info
            save_model_info(run.info.run_id, 'lgbm_model', 'experiment_info.json')

            mlflow.log_artifact(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Evaluate model and get metrics
            report, cm = evaluate_model(model, X_test_trf, y_test)

            # Log classification report metrics for the test data
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    })

            # log confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, cmap="Blues", fmt='d')
            plt.title('confusion matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            cm_file_path = f'confusion_matrix.png'
            plt.savefig(cm_file_path)
            mlflow.log_artifact(cm_file_path)
            plt.close()

            # Add important tags
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")

        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")

    
if __name__ == '__main__':
    main()