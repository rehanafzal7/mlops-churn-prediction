import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test, params):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.params = params
        self.model = None
    
    def train_and_log(self):
        """Trains the model and logs parameters/metrics to MLflow."""
        # Ensure MLflow tracking URI is set (e.g., to a local 'mlruns' folder)
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri("file:./mlruns")
        
        with mlflow.start_run(run_name="Logistic_Regression_Churn_Prediction"):
            # 1. Log parameters
            mlflow.log_params(self.params)
            print(f"MLflow Parameters: {self.params}")
            
            # 2. Train model
            self.model = LogisticRegression(**self.params)
            self.model.fit(self.X_train, self.y_train)
            print("Model training complete.")
            
            # 3. Evaluate model
            predictions = self.model.predict(self.X_test)
            probabilities = self.model.predict_proba(self.X_test)[:, 1]
            
            accuracy = accuracy_score(self.y_test, predictions)
            precision = precision_score(self.y_test, predictions)
            recall = recall_score(self.y_test, predictions)
            f1 = f1_score(self.y_test, predictions)
            roc_auc = roc_auc_score(self.y_test, probabilities)
            
            # 4. Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)
            print(f"MLflow Metrics: Accuracy={accuracy:.4f}, ROC AUC={roc_auc:.4f}")
            
            # 5. Log model artifact
            mlflow.sklearn.log_model(self.model, "model", registered_model_name="ChurnPredictionModel")
            print("Model artifact logged to MLflow.")
            
            # You can also log feature names for better model understanding
            # if hasattr(self.X_train, 'columns'):
            #     mlflow.log_param("features", self.X_train.columns.tolist())
        
        return self.model