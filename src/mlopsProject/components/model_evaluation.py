import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report,ConfusionMatrixDisplay, \
                            precision_score, recall_score, f1_score, roc_auc_score,roc_curve 
from sklearn.metrics import roc_auc_score,roc_curve
from mlopsProject.utils.common import save_json
from urllib.parse import urlparse
import numpy as np
import joblib
from mlopsProject.entity.config_entity import ModelEvaluationConfig
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
    
    def evaluate_clf(true, predicted):
        acc = accuracy_score(true, predicted) # Calculate Accuracy
        f1 = f1_score(true, predicted) # Calculate F1-score
        precision = precision_score(true, predicted) # Calculate Precision
        recall = recall_score(true, predicted)  # Calculate Recall
        roc_auc = roc_auc_score(true, predicted) #Calculate Roc
        return acc, f1 , precision, recall, roc_auc
    

    def save_results(self):
        models_list = []
        accuracy_list = []
        auc= []

        model = joblib.load(self.config.model_path)

        test_x = pd.read_csv(self.config.X_test_data_path)
        test_y = pd.read_csv(self.config.y_test_data_path)
        
        predicted_qualities = model.predict(test_x)

        accuracy_sc = accuracy_score(test_y,predicted_qualities)
        cr = classification_report(test_y,predicted_qualities)

         # Saving metrics as local
        # scores = {"accuracy": accuracy_sc, "cr": cr}
        scores = {"accuracy": accuracy_sc}
        save_json(path=Path(self.config.metric_file_name), data=scores)
