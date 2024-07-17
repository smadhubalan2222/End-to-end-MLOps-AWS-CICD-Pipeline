import pandas as pd
import os
from mlopsProject import logger
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib
from mlopsProject.entity.config_entity import ModelTrainerConfig



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        
    
    def train(self):
        train_x = pd.read_csv(self.config.X_train_data_path)
        test_x = pd.read_csv(self.config.X_test_data_path)
        train_y = pd.read_csv(self.config.y_train_data_path)
        test_y = pd.read_csv(self.config.y_test_data_path)

        best_model = XGBClassifier(min_child_weight=self.config.min_child_weight, max_depth=self.config.max_depth, n_jobs=-1)
        best_model = best_model.fit(train_x, train_y)

        joblib.dump(best_model, os.path.join(self.config.root_dir, self.config.model_name))