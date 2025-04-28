"""Evaluation script for measuring mean squared error."""
import json
import logging
import pathlib
import pickle
import tarfile

import numpy as np
import pandas as pd
import pickle
import joblib

from sklearn.metrics import accuracy_score,classification_report

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())



if __name__ == "__main__":
    logger.debug("Starting evaluation.")
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    logger.debug("Loading model.")
    model = joblib.load('./model.h')

    logger.debug("Reading test data.")
    test_path = "/opt/ml/processing/test"

    x_test = pd.read_csv(os.path.join(test_path, 'x_test.csv'))
    y_test = pd.read_csv(os.path.join(test_path, 'y_test.csv'))

    logger.info("Performing predictions against test data.")
    y_pred = model.predict(x_test)

    logger.debug("Calculating accuracy.")
    accuracy = accuracy_score(y_test, y_pred)
    report_dict = {
        "binary_classification_metrics": {
            "accuracy": {
                "value": accuracy,
                "standard_deviation": "NaN"
            },
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out evaluation report with mse: %f", mse)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
