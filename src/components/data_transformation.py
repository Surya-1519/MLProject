import sys,os
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer    ## To handle missing values
from sklearn.pipeline import Pipeline       ## To implement pipelines


from dataclasses import dataclass

@dataclass

class data_transformation_config():
    preprocessor_obj_file_path = os.path.join("Artifacts","preprocessor.pkl")

class data_transformation():
    def __init__(self):
        self.data_transformation_config = data_transformation_config()
    
    def get_data_transformer_object(self):
       try:
           numerical_features = ['reading_score', 'writing_score']
           categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

           #To handle missing features, we have to make pipelines
           num_pipeline = Pipeline(
               steps = [
                   ("imputer",SimpleImputer(strategy="mean")),
                   ("scaler",StandardScaler())
               ]
           )
           cat_pipeline = Pipeline(
               steps = [
                   ("imputer",SimpleImputer(strategy="most_frequent")),
                   ("oh_encoder",OneHotEncoder()),
                   ("scaler",StandardScaler(with_mean=False))
               ]
           )
           logging.info(f"Categorical features: {categorical_features}")
           logging.info(f"Numerical features: {numerical_features}")

           preprocessor = ColumnTransformer(
               [
                   ("num_pipeline",num_pipeline,numerical_features),
                   ("cat_pipeline",cat_pipeline,categorical_features)
               ]

           )
           return preprocessor

       except Exception as e:
           raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data read")
            logging.info("Getting preprocessor....")

            preprocessing_obj = self.get_data_transformer_object()
            target_feature = "math_score"
            numerical_features = ['reading_score', 'writing_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            
            input_feature_train_df = train_df.drop(columns=target_feature,axis=1)
            target_feature_train_df = train_df[target_feature]

            input_feature_test_df = test_df.drop(columns=target_feature,axis=1)
            target_feature_test_df = test_df[target_feature]

            logging.info("Applying preprocessing object on train and test data....")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(file_path = self.data_transformation_config.preprocessor_obj_file_path,obj = preprocessing_obj)

            return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)
        
        except Exception as e:
            raise CustomException(e,sys)

       
    


    

           



