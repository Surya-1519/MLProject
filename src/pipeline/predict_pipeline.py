import os,sys,pickle
from src.logger import logging
from src.exception import CustomException

import numpy as np
import pandas as pd
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = 'Artifacts\model.pkl'
            preprocessor_path = 'Artifacts\preprocessor.pkl'
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            scaled_input_data = preprocessor.transform(features)
            y_pred = model.predict(scaled_input_data)

            return y_pred
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,reading_score:int,writing_score:int,gender:str,race_ethnicity:str,parental_level_of_education:str,lunch:str,test_preparation_course:str):
        self.reading_score = reading_score
        self.writing_score = writing_score
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

