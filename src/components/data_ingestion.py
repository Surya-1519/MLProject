import os, sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass

class data_ingestion_config:
    test_data_path = str=os.path.join("Artifacts","test.csv")
    train_data_path = str=os.path.join("Artifacts","train.csv")
    raw_data_path = str=os.path.join("Artifacts","raw_data.csv")

class data_ingestion:
    def __init__(self):
        self.data_ingestion_config = data_ingestion_config()
    
    def initiate_data_ingestion(self):
        logging.info("Entering data ingestion")
        try:
            df = pd.read_csv("notebooks/data/stud.csv")
            logging.info("Read data successfully into dataframe")
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Raw data saved. Train test split initiated")
            
            train_data,test_data = train_test_split(df,test_size=0.2,random_state=42)
            train_data.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data ingestion completed, test and train data splitted")

            return (self.data_ingestion_config.test_data_path,self.data_ingestion_config.train_data_path)

        except Exception as e:
            raise CustomException(e,sys)


if __name__=="__main__":
    obj = data_ingestion()
    obj.initiate_data_ingestion()



