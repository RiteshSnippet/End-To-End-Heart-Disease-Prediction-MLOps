from heart_disease.components.data_ingestion import DataIngestion
from heart_disease.components.data_transformation import DataTransformation
from heart_disease.components.model_trainer import ModelTrainer
from heart_disease.components.model_evaluation import ModelEvaluation


def run_training_pipeline():

    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
    
    print(f"Train data path: {train_data_path}")
    print(f"Test data path: {test_data_path}")


    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initialize_data_transformation(train_data_path, test_data_path)

    print(f"Preprocessor saved at: {preprocessor_path}")


    model_trainer = ModelTrainer()
    best_model_name = model_trainer.initiate_model_training(train_arr, test_arr)
    
    print(f"Best Model: {best_model_name}")


    model_evaluation = ModelEvaluation()
    model_evaluation.initiate_model_evaluation(test_arr, best_model_name)

    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    run_training_pipeline()