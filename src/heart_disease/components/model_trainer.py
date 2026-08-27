import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from src.heart_disease.logger import logging
from src.heart_disease.exception import CustomException
from src.heart_disease.utils.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts","Model.joblib")

class ModelTrainer:

    def __init__(self):

        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):

        try:

            logging.info("Starting model training")

            target_column = "target"

            X_train = train_array.drop(columns=[target_column])
            y_train = train_array[target_column]

            X_test = test_array.drop(columns=[target_column])
            y_test = test_array[target_column]

            logging.info("Independent and dependent variables, split successfully")
            logging.info(f"X_train shape: {X_train.shape}")
            logging.info(f"X_test shape: {X_test.shape}")
            logging.info(f"y_train shape: {y_train.shape}")
            logging.info(f"y_test shape: {y_test.shape}")

            models = {

                "Logistic Regression":
                    LogisticRegression(
                        max_iter=2000,
                        random_state=42
                    ),

                "Random Forest Classifier":
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        max_depth=5,
                        min_samples_leaf=2,
                        n_jobs=-1
                    ),

                "XGBoost":
                    XGBClassifier(
                        learning_rate=0.03,
                        n_estimators=200,
                        max_depth=3,
                        gamma=0.2,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        eval_metric="logloss",
                        random_state=42,
                        min_child_weight=3,
                        reg_lambda=2,
                        n_jobs=-1
                    ),

                "Decision Tree":
                    DecisionTreeClassifier(
                        criterion="entropy",
                        random_state=42,
                        max_depth=6,
                        min_samples_leaf=3
                    ),

                "Support Vector Machine":
                    SVC(
                        kernel="rbf",
                        C=2,
                        probability=True,
                        random_state=42
                    )
            }

            logging.info(f"Models selected: {list(models.keys())}")

            model_report, trained_models = evaluate_model(X_train, y_train, X_test, y_test, models)
            logging.info(f"Model evaluation report: {model_report}")

            print("\n" + "=" * 90)
            print("MODEL EVALUATION RESULTS")
            print("=" * 90)

            for model_name, metrics in model_report.items():
                print(f"\nModel: {model_name}")
                print(f"Accuracy : {metrics['accuracy']:.4f}")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall   : {metrics['recall']:.4f}")
                print(f"F1 Score : {metrics['f1_score']:.4f}")

                if metrics["roc_auc"] is not None:
                    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")

                if metrics["pr_auc"] is not None:
                    print(f"PR-AUC   : {metrics['pr_auc']:.4f}")

            best_model_name = max(
                model_report,
                key=lambda x:
                model_report[x]["f1_score"]
            )

            best_model_metrics = model_report[best_model_name]
            best_model = trained_models[best_model_name]
            
            print("\n" + "=" * 90)
            print(f"BEST MODEL: {best_model_name}")

            print(
                f"Accuracy : "
                f"{best_model_metrics['accuracy']:.4f}"
            )
            print(
                f"Precision: "
                f"{best_model_metrics['precision']:.4f}"
            )
            print(
                f"Recall   : "
                f"{best_model_metrics['recall']:.4f}"
            )
            print(
                f"F1 Score : "
                f"{best_model_metrics['f1_score']:.4f}"
            )
            print(
                f"ROC-AUC  : "
                f"{best_model_metrics['roc_auc']:.4f}"
            )
            print(
                f"PR-AUC   : "
                f"{best_model_metrics['pr_auc']:.4f}"
            )
            print("=" * 90)

            logging.info(
                f"""
                Best Model : {best_model_name}
                Accuracy   : {best_model_metrics['accuracy']:.4f}
                Precision  : {best_model_metrics['precision']:.4f}
                Recall     : {best_model_metrics['recall']:.4f}
                F1 Score   : {best_model_metrics['f1_score']:.4f}
                ROC-AUC    : {best_model_metrics['roc_auc']}
                PR-AUC     : {best_model_metrics['pr_auc']}
                """
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Best model saved successfully")
            return best_model_name

        except Exception as e:
            logging.error("Exception occurred during model training")
            raise CustomException(e, sys)