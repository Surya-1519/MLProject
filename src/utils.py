import os,sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dirname = os.path.dirname(file_path)
        os.makedirs(dirname,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            
            model.fit(X_train,y_train)

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            train_model_score = r2_score(y_train,y_pred_train)
            test_model_score = r2_score(y_test,y_pred_test)

            report[list(models.keys())[i]] = test_model_score
        
        return report

    except Exception as e:
        raise CustomException(e,sys)

