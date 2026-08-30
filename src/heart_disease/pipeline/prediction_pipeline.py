import os
import sys
import pandas as pd
from src.heart_disease.logger import logging
from src.heart_disease.utils.utils import load_object
from src.heart_disease.exception import CustomException


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):

        try:

            logging.info("Starting prediction pipeline")

            preprocessor_path = os.path.join("artifacts", "Preprocessor.joblib")
            model_path = os.path.join("artifacts", "Model.joblib")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            logging.info("Preprocessor and model loaded successfully")

            transformed_data = preprocessor.transform(features)
            transformed_data = pd.DataFrame(
                transformed_data,
                columns=features.columns,
                index=features.index
            )
            logging.info("Input features transformed successfully")

            prediction = model.predict(transformed_data)

            logging.info(f"Prediction generated: {prediction}")

            prediction_probability = None

            if hasattr(model, "predict_proba"):
                prediction_probability = (
                    model.predict_proba(transformed_data)[:, 1]
                )

                logging.info(
                    f"Prediction probability: "
                    f"{prediction_probability}"
                )

            return (
                prediction,
                prediction_probability
            )

        except Exception as e:
            logging.error("Exception occurred in prediction pipeline")
            raise CustomException(e, sys)


class CustomData:

    def __init__(
        self,
        age: int,
        sex: int,
        cp: int,
        trestbps: int,
        chol: int,
        fbs: int,
        restecg: int,
        thalach: int,
        exang: int,
        oldpeak: float,
        slope: int,
        ca: int,
        thal: int
    ):

        self.age = age
        self.sex = sex
        self.cp = cp
        self.trestbps = trestbps
        self.chol = chol
        self.fbs = fbs
        self.restecg = restecg
        self.thalach = thalach
        self.exang = exang
        self.oldpeak = oldpeak
        self.slope = slope
        self.ca = ca
        self.thal = thal

    def get_data_as_dataframe(self):

        try:

            custom_data_input_dict = {

                "age": [self.age],

                "sex": [self.sex],

                "cp": [self.cp],

                "trestbps": [self.trestbps],

                "chol": [self.chol],

                "fbs": [self.fbs],

                "restecg": [self.restecg],

                "thalach": [self.thalach],

                "exang": [self.exang],

                "oldpeak": [self.oldpeak],

                "slope": [self.slope],

                "ca": [self.ca],

                "thal": [self.thal]
            }

            df = pd.DataFrame(custom_data_input_dict)

            logging.info("Prediction dataframe created successfully")
            logging.info(f"Prediction dataframe:\n{df}")

            return df

        except Exception as e:
            logging.error("Error while creating prediction dataframe")
            raise CustomException(e, sys)