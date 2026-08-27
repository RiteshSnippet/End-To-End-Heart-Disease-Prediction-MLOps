import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.heart_disease.logger import logging
from src.heart_disease.exception import CustomException

@dataclass
class DataIngestionConfig:

    raw_data_path: str = os.path.join("artifacts","raw_data.csv")
    train_data_path: str = os.path.join("artifacts","train_data.csv")
    test_data_path: str = os.path.join("artifacts","test_data.csv")

class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")

        try:
            data_path = os.path.join("notebooks","Data","heart.csv")
            data = pd.read_csv(data_path)

            logging.info(f"Heart disease dataset loaded successfully from: {data_path}")
            logging.info(f"Dataset shape: {data.shape}")

            os.makedirs(
                os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True
            )

            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("Raw data file created successfully")

            target_column = "target"

            if target_column not in data.columns:
                raise ValueError(
                    f"Target column '{target_column}' "
                    f"not found in dataset."
                )

            logging.info(f"Target column '{target_column}' found successfully")

            logging.info("Splitting data into training and testing datasets")

            y = data[target_column]

            train_data, test_data = train_test_split(
                data,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            logging.info("Data splitting completed successfully")
            logging.info(f"Training dataset shape: {train_data.shape}")
            logging.info(f"Testing dataset shape: {test_data.shape}")

            train_data.to_csv(
                self.ingestion_config.train_data_path,
                index=False
            )

            test_data.to_csv(
                self.ingestion_config.test_data_path,
                index=False
            )

            logging.info("Training and testing datasets created successfully")
            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Exception occurred during data ingestion")
            raise CustomException(e, sys)