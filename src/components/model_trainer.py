import os,sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_models

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score,mean_squared_error

@dataclass

class ModelTrainerConfig():
    trained_model_file_path = os.path.join("Artifacts","model.pkl")

class ModelTraier():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Splitting train and test data...")
            X_train,y_train,X_test,y_test = (train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])
            models = {
                "Linear Regression":LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "XGBoost": XGBRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Adaboost": AdaBoostRegressor(),
                "Cataboosting Regressor": CatBoostRegressor(verbose=False),
                "K-Nearest Neighbor": KNeighborsRegressor(),
                "Gradient Boost Regressor": GradientBoostingRegressor()
            }
            models_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_score = max(sorted(list(models_report.values())))

            for model_name, score in models_report.items():
                if score == best_model_score:
                   best_model_name = model_name
                   break

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No Best model found!")
            
            logging.info("Found best model for training and test data")

            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)

            logging.info("Saved the best model")

            
            return best_model_score
            

        except Exception as e:
            raise CustomException(e,sys)





