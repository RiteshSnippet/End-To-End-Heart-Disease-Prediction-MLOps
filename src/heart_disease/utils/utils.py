import os
import sys
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from src.heart_disease.logger import logging
from src.heart_disease.exception import CustomException


def save_object(file_path, obj):

    try:

        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        joblib.dump(obj,file_path)

        logging.info(f"Object saved successfully at: {file_path}")

    except Exception as e:
        logging.error("Exception occurred in save_object function")
        raise CustomException(e, sys)


def load_object(file_path):

    try:

        logging.info(f"Loading object from: {file_path}")

        obj = joblib.load(file_path)

        logging.info("Object loaded successfully")
        return obj

    except Exception as e:
        logging.error("Exception occurred in load_object function")
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models):

    try:

        report = {}
        trained_models = {}

        logging.info("Starting model evaluation")

        for model_name, model in models.items():

            logging.info(f"Training model: {model_name}")

            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)

            y_test_prob = None

            if hasattr(model, "predict_proba"):
                y_test_prob = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_test_pred)
            precision = precision_score(
                y_test,
                y_test_pred,
                zero_division=0
            )
            recall = recall_score(
                y_test,
                y_test_pred,
                zero_division=0
            )
            f1 = f1_score(
                y_test,
                y_test_pred,
                zero_division=0
            )

            roc_auc = None

            if y_test_prob is not None:
                roc_auc = roc_auc_score(
                    y_test,
                    y_test_prob
                )

            pr_auc = None

            if y_test_prob is not None:

                pr_auc = average_precision_score(
                    y_test,
                    y_test_prob
                )


            report[model_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc
            }

            trained_models[model_name] = model

            logging.info(
                f"""
                Model: {model_name}
                Accuracy: {accuracy:.4f}
                Precision: {precision:.4f}
                Recall: {recall:.4f}
                F1 Score: {f1:.4f}
                ROC-AUC: {roc_auc}
                PR-AUC: {pr_auc}
                """
            )

        logging.info("All models trained and evaluated successfully")

        return (
            report,
            trained_models
        )

    except Exception as e:
        logging.error("Exception occurred during model evaluation")
        raise CustomException(e, sys)