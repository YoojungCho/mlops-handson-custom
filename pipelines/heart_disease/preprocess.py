"""Feature engineers the abalone dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/heart_statlog_cleveland_hungary_final.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.debug("Reading downloaded data.")
    data = pd.read_csv(fn)
    os.unlink(fn)

    # Preprocessing
    x = data.drop('target', axis=1)
    y = data['target']

    logger.info("Splitting %d rows of data into train and test datasets.", len(x))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    logger.info("Writing out datasets to %s.", base_dir)
    pd.DataFrame(x_train).to_csv(f"{base_dir}/train/x_train.csv", header=False, index=False)
    pd.DataFrame(y_train).to_csv(f"{base_dir}/train/y_train.csv", header=False, index=False)
    pd.DataFrame(x_test).to_csv(f"{base_dir}/test/x_test.csv", header=False, index=False)
    pd.DataFrame(y_test).to_csv(f"{base_dir}/test/y_test.csv", header=False, index=False)
