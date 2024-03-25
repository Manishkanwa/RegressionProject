import os, sys

import numpy as np
import pandas as pd

import pickle
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def save_object(file_path, obj):
    try: 
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj, file_obj)
            
        
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as obj :
            return pickle.load(obj)
    except Exception as e:
        logging.info("exception occured in loading object!")
        raise CustomException(e, sys)

def evaluate_module(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train models
            model.fit(X_train, y_train)

            #predict test data  
            y_test_pred = model.predict(X_test)

            test_model_score = r2_score(y_true=y_test, y_pred= y_test_pred)
            report[list(models.keys())[i]] = test_model_score

            return report

    except Exception as e:
        logging.info("Exception occured during model training")
        raise CustomException(e, sys)