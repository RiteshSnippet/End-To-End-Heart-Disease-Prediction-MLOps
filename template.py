from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

package="heart_disease" 

list_of_files = [
    ".github/workflows/",
    "notebooks/Data/.gitkeep",
    f"src/{package}/__init__.py",
    f"src/{package}/exception.py",
    f"src/{package}/logger.py",
    f"src/{package}/utils/__init__.py",
    f"src/{package}/utils/utils.py",
    f"src/{package}/components/__init__.py",
    f"src/{package}/components/data_ingestion.py",
    f"src/{package}/components/data_transformation.py",
    f"src/{package}/components/model_trainer.py",
    f"src/{package}/pipeline/__init__.py",
    f"src/{package}/pipeline/prediction_pipeline.py",
    f"src/{package}/pipeline/training_pipeline.py",
    "static/styles.css",
    "templates/home.html",
    "app.py",
    "Dockerfile",
]

def create_project_structure() -> None:
    for file_path in list_of_files:
        path = Path(file_path)

        path.parent.mkdir(parents=True, exist_ok=True)

        if not path.exists() or path.stat().st_size == 0:
            path.touch()
            logging.info("Created: %s", path)
        else:
            logging.info("Already exists: %s", path)


if __name__ == "__main__":
    create_project_structure()
    logging.info("Project structure created successfully!")