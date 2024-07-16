import os
from mlopsProject import logger
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek, SMOTEENN
import pandas as pd
from mlopsProject.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)
        data.drop(columns=["product_var_3", "marital_status", "occupation", "location","family_history_3","employment_type"], inplace=True, axis=1)
        X = data.drop('claim_status', axis=1)
        y = data['claim_status']
        smt = SMOTEENN(random_state=42,sampling_strategy='minority')
        X_res, y_res = smt.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_res,y_res,test_size=0.2,random_state=42)

        X_train.to_csv(os.path.join(self.config.root_dir, "X_train.csv"),index = False)
        X_test.to_csv(os.path.join(self.config.root_dir, "X_test.csv"),index = False)

        y_train.to_csv(os.path.join(self.config.root_dir, "y_train.csv"),index = False)
        y_test.to_csv(os.path.join(self.config.root_dir, "y_test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(X_train.shape)
        logger.info(X_test.shape)
        logger.info(y_train.shape)
        logger.info(y_test.shape)

        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)    
