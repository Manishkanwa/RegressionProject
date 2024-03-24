import os, sys

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    test_data_path = os.path.join("artifacts","test.csv")
    train_data_path = os.path.join("artifacts", "train.csv")
    raw_data_path = os.path.join("artifacts", "raw.csv")

class DataIngestion :
    def __init__(self):
        self.Ingestion_Config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion methods Starts")
        try:
            df = pd.read_csv("notebooks\data\data.csv")
            logging.info("Data read as pandas DataFrame")

            os.makedirs(os.path.dirname(self.Ingestion_Config.raw_data_path), exist_ok=True)
            
            df.to_csv(self.Ingestion_Config.raw_data_path,index = False)
            logging.info("Train Test Split")
            train_set, test_set = train_test_split(df, test_size=0.30)

            train_set.to_csv(self.Ingestion_Config.train_data_path,index = False, header = True)
            test_set.to_csv(self.Ingestion_Config.test_data_path,index = False, header = True)
            logging.info("Ingestion of Data is completed")
            return (
                self.Ingestion_Config.train_data_path, 
                self.Ingestion_Config.test_data_path
                )
            
        except Exception as e :
            logging.info("Exception occured at data ingestion stage")
            raise CustomException(e, sys)
        






        