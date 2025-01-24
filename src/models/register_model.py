import mlflow
import logging
import json
import os

import mlflow.client

# logging configuration
logger = logging.getLogger('register_model')
logger.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler('errors.log')
fileHandler.setLevel(logging.ERROR)

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fileHandler.setFormatter(formatter)
consoleHandler.setFormatter(formatter)

logger.addHandler(fileHandler)
logger.addHandler(consoleHandler)


# mlflow tracking uri
mlflow.set_tracking_uri()


def load_model_info(path: str) -> dict:
    """Load the model info from given json file"""
    try:
        with open(path, 'r') as file:
            info = json.load(file)
        logger.debug('model info loaded successfully')

        return info
    except FileNotFoundError:
        logger.error('file not found error: %s', path)
        raise
    except Exception as e:
        logger.error('Error occurred while loading the model info: %s', e)
        raise


def register_model(model_name: str, model_info: dict) -> None:
    """Register the model to the mlflow model registry"""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
        # register model
        version = mlflow.register_model(model_uri, model_name)

        # transition the model to staging
        client = mlflow.client.MlflowClient()
        client.transition_model_version_stage(model_name, version.version, 'Staging')
        logger.debug('model transitioned to staging')
    except Exception as e:
        logger.error('error while model registration: %s', e)
        raise


def main():
    try:
        # load model info
        model_info_path = 'experiment_info.json'
        model_info = load_model_info(model_info_path)

        # register model
        model_name = 'lgbm_model'
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete model registration stage: %s', e)


if __name__ == '__main__':
    main()