"""Feature engineers the abalone dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
import pandas as pd
import pickle
import joblib

from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.debug("Reading train data.")
    train_path = "/opt/ml/input/data/train"

    file_list = os.listdir(train_path)
    print ("train file_list: {}".format(file_list))
    
    x_train = pd.read_csv(os.path.join(train_path, 'x_train.csv'))
    y_train = pd.read_csv(os.path.join(train_path, 'y_train.csv'))
    
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    
    joblib.dump(rfc, '/opt/ml/model/model.h')
    