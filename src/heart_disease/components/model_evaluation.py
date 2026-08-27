import os
import sys
import mlflow
import mlflow.sklearn
import dagshub
from dotenv import load_dotenv
from urllib.parse import urlparse
from src.heart_disease.exception import CustomException
from src.heart_disease.logger import logging
from src.heart_disease.utils.utils import load_object
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

class ModelEvaluation:

    def __init__(self):
        pass


    def eval_metrics(self, actual, pred, pred_proba=None):

        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, zero_division=0)
        recall = recall_score(actual, pred, zero_division=0)
        f1 = f1_score(actual, pred, zero_division=0)

        roc_auc = None
        pr_auc = None

        if pred_proba is not None:
            roc_auc = roc_auc_score(actual, pred_proba)
            pr_auc = average_precision_score(actual,pred_proba)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        }


    def initiate_model_evaluation(self, test_array, best_model_name):

        try:

            logging.info("Starting model evaluation")

            target_column = "target"

            X_test = test_array.drop(columns=[target_column])
            y_test = test_array[target_column]

            logging.info(f"Test features shape: {X_test.shape}")
            logging.info(f"Test target shape: {y_test.shape}")

            model_path = os.path.join("artifacts", "Model.joblib")
            model = load_object(model_path)

            logging.info("Trained model loaded successfully")


            load_dotenv()

            dagshub.init(
                repo_owner="RiteshSnippet",
                repo_name="End-To-End-Heart-Disease-Prediction-MLOps",
                mlflow=True
            )

            tracking_url_type_store = (
                urlparse(mlflow.get_tracking_uri()).scheme
            )

            print(
                "MLflow Tracking URI:",
                mlflow.get_tracking_uri()
            )

            print(
                "Tracking Type:",
                tracking_url_type_store
            )

            logging.info("Evaluating model on test dataset")

            with mlflow.start_run():
                mlflow.log_param(
                    "model_name",
                    best_model_name
                )

                model_params = model.get_params()

                mlflow.log_params(
                    model_params
                )
                logging.info(f"Logged parameters for {best_model_name}")

                y_pred = model.predict(X_test)

                y_pred_prob = None

                if hasattr(model, "predict_proba"):
                    y_pred_prob = model.predict_proba(X_test)[:, 1]


                metrics = self.eval_metrics(y_test, y_pred, y_pred_prob)

                print("\n" + "=" * 70)
                print("FINAL MODEL EVALUATION")
                print("=" * 70)

                print(f"Best Model : {best_model_name}")
                print(f"Accuracy   : {metrics['accuracy']:.4f}")
                print(f"Precision  : {metrics['precision']:.4f}")
                print(f"Recall     : {metrics['recall']:.4f}")
                print(f"F1 Score   : {metrics['f1_score']:.4f}")

                if metrics["roc_auc"] is not None:
                    print(
                        f"ROC-AUC    : "
                        f"{metrics['roc_auc']:.4f}"
                    )

                if metrics["pr_auc"] is not None:
                    print(
                        f"PR-AUC     : "
                        f"{metrics['pr_auc']:.4f}"
                    )
                print("=" * 70)

                mlflow.log_metric(
                    "Accuracy",
                    metrics["accuracy"]
                )

                mlflow.log_metric(
                    "Precision",
                    metrics["precision"]
                )

                mlflow.log_metric(
                    "Recall",
                    metrics["recall"]
                )

                mlflow.log_metric(
                    "F1 Score",
                    metrics["f1_score"]
                )

                if metrics["roc_auc"] is not None:

                    mlflow.log_metric(
                        "ROC-AUC",
                        metrics["roc_auc"]
                    )

                if metrics["pr_auc"] is not None:

                    mlflow.log_metric(
                        "PR-AUC",
                        metrics["pr_auc"]
                    )

                if tracking_url_type_store != "file":

                    mlflow.sklearn.log_model(
                        model,
                        artifact_path="Model",
                        registered_model_name=best_model_name
                    )

                else:

                    mlflow.sklearn.log_model(
                        model,
                        artifact_path="Model"
                    )

                logging.info("Model and metrics logged to MLflow successfully")

        except Exception as e:
            logging.error("Exception occurred during model evaluation")
            raise CustomException(e, sys)