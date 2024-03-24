import pandas as pd
import numpy as np
import sys
import os

from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_module
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

@dataclass
class ModeltrainerConfig :
    trained_model_file_path = os.path.join("artifacts" , "model.pkl")

class ModelTrainer :
    def __init__(self):
        self.model_trainer_config = ModeltrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting dependent and independent variables from train and test array")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:,-1],
                test_array[:, :-1],
                test_array[:,-1]
            )
            ## Train multiple models
            models = {
                "LinearRegression" : LinearRegression(),
                "Lasso" : Lasso(),
                "Ridge" : Ridge(),
                "ElasticNet" : ElasticNet() 
            }

            model_report: dict = evaluate_module(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print("\n", "="*35)
            logging.info(f"Model report : {model_report}")

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]    

            best_model = models[best_model_name] 
            print(f"best model name : {best_model_name}, R2 score :{best_model_score} ")
            print("\n", "="*35)
            logging.info(f"best model name : {best_model_name}, R2 score :{best_model_score} ")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
                )
            
        except Exception as e:
            logging.info("Error in training the model.") 
            raise CustomException(e,sys)
