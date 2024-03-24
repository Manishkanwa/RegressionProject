import pandas as pd
import numpy as np
import sys
from dataclasses import dataclass
from src.utils import save_object
from sklearn import impute, preprocessing, compose, pipeline
import os
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataTransformationconfig:
    preprocessor_object_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()
        logging.info("data transform config")
    def get_data_transformation_object(self) -> preprocessing :
        try:
            logging.info("Data Transformation has started!")
            
            # These are the columns that should be ordinal encoded
            X_categorical = ['cut', 'color', 'clarity']
            X_numeric = ['carat', 'depth', 'table', 'x', 'y', 'z']

            
            #These are the ranks for the categorical columns
            cut_categories = ['Fair', 'Good', 'Very Good', 'Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1', 'SI2', 'SI1','VS2' , 'VS1', 'VVS2', 'VVS1', 'IF']

            logging.info("Pipeline initiated!!")
            #creating the pipeline for transformation
            num_pipline = pipeline.Pipeline(
                steps=[
                ("imputer", impute.SimpleImputer(strategy="median") ),
                ("scaler", preprocessing.StandardScaler())
                ]
            )

            cat_pipline = pipeline.Pipeline(
                steps=[
                    ("imputer", impute.SimpleImputer(strategy="most_frequent")),
                    ("ordinal_encoder", preprocessing.OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                    ("scaler", preprocessing.StandardScaler())
                ])

            preprocessor = compose.ColumnTransformer([
                ("num_pipline", num_pipline, X_numeric),
                ("cat_pipline", cat_pipline, X_categorical)
            ])
            logging.info("Pipeline completed !!")

            return preprocessor
            
        except Exception as e:
            logging.info()
            raise CustomException(e,sys)

    def initiate_data_transformation(self, train, test_path):
        try :
            train_data = pd.read_csv(train)
            test_data = pd.read_csv(test_path)
            logging.info("read train and test data complete !!")
            logging.info(f"Train data is :-\n {train_data.head().to_string()}")
            logging.info(f"Test data is :-\n {test_data.head().to_string()}")

            logging.info("Getting data transformation object")

            
            preprocessing_obj = self.get_data_transformation_object()
            
            target_feature = "price"
            drop_columns = [target_feature, "id"]
            
            input_features_train_df = train_data.drop(columns=drop_columns, axis=1)
            target_features_train_df = train_data[target_feature]
            
            input_features_test_df = test_data.drop(columns= drop_columns, axis = 1)
            target_features_test_df = test_data[target_feature]

            logging.info("applying preprocessing on training and test dataset")

            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df)

            train_arr = np.c_[input_features_train_arr, np.array(target_features_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_features_test_df)]

            save_object(
                file_path= self.data_transformation_config.preprocessor_object_file_path,
                obj= preprocessing_obj
            )
            logging.info("Preprocessor pickle file saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_object_file_path
            )

        except Exception as e:
            logging.info("Exception occured in the initiate of data transformation")
        