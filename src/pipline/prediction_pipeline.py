import sys 
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predictpipeline(self,features):
        try:
            model_file_path = os.path.join("artifacts", "model.pkl")
            preprocessor_file_path = os.path.join("artifacts", "preprocessor.pkl")
            model = load_object(model_file_path)
            preprocessor = load_object(preprocessor_file_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
            
        except Exception as e:
            logging.info("ecception occured in prediction")
            raise CustomException(e,sys)

class custom_data:
    def __init__(self, carat:float,
                 depth : float,
                 table : float,
                 x : float,
                 y : float,
                 z : float,
                 cut : str,
                 color : str,
                 clarity : str,) -> None:
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x 
        self.y = y 
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        try:
            data = {'carat' :[self.carat], 
                    'depth' : [self.depth],
                    'table' : [self.table],
                    'x' : [self.x],
                    'y' : [self.y],
                    'z' : [self.z],
                    'cut' : [self.cut],
                    'color' : [self.color],
                    'clarity' : [self.clarity]}
            df = pd.DataFrame(data )
            logging.info("dataframe Gathered")
            return df
        except Exception as e:
            logging.info("Exception occured in prediction pipline")
            raise CustomException(e,sys)