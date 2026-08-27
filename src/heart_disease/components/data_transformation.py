import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.heart_disease.logger import logging
from src.heart_disease.exception import CustomException
from src.heart_disease.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "Preprocessor.joblib")

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):

        try:

            logging.info("Data Transformation initiated")

            numerical_cols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
            logging.info(f"Features to be scaled: {numerical_cols}")

            numerical_pipeline = Pipeline(
                steps=[
                    (
                        "standard_scaler",
                        StandardScaler()
                    )
                ]
            )

            logging.info("StandardScaler pipeline created")

            preprocessor = numerical_pipeline
            logging.info("Preprocessor created successfully")

            return preprocessor

        except Exception as e:
            logging.error("Exception occurred in get_data_transformation")
            raise CustomException(e, sys)

    def initialize_data_transformation(self, train_path, test_path):
        
        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Training and testing data loaded successfully")
            logging.info(f"Training data shape: {train_df.shape}")
            logging.info(f"Testing data shape: {test_df.shape}")

            preprocessing_obj = self.get_data_transformation()

            target_column_name = "target"

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Input and target features separated successfully")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("StandardScaler applied successfully")

            feature_names = input_feature_train_df.columns
            
            train_arr = pd.DataFrame(
                input_feature_train_arr,
                columns=feature_names,
                index=train_df.index
            )

            test_arr = pd.DataFrame(
                input_feature_test_arr,
                columns=feature_names,
                index=test_df.index
            )

            train_arr[target_column_name] = (
                target_feature_train_df.values
            )

            test_arr[target_column_name] = (
                target_feature_test_df.values
            )

            logging.info("Target column added successfully")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessor object saved successfully")
            logging.info("Data transformation completed successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error("Exception occurred during data transformation")
            raise CustomException(e, sys)