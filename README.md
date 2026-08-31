# End-To-End-Heart-Disease-Prediction-MLOps

A production-oriented machine learning system for predicting heart disease risk, built end-to-end with a full MLOps stack: a modular training pipeline, a Flask inference service, automated tests, and CI/CD, deployed live on Render.

🔗 **Live Demo:** [heart-disease-prediction-1lrm.onrender.com](https://heart-disease-prediction-1lrm.onrender.com)

## Overview

Predicting heart disease from clinical data is a classic healthcare ML problem, but going from a notebook to something people can actually use is a different challenge. This project treats it as a full lifecycle problem: data ingestion, preprocessing, model training, evaluation, and a served prediction endpoint — packaged so it's reproducible, testable, and deployable rather than living only in a Jupyter notebook.

## Key Features

- **Modular pipeline** for data ingestion, preprocessing, model training, and evaluation, organized as an installable Python package
- **Flask web application** with a simple form-based UI for entering patient data and getting a prediction
- **Automated tests** (pytest) covering the prediction path
- **Fast, reproducible environment management with `uv`**, backed by a locked dependency file
- **CI/CD pipeline** via GitHub Actions
- **Live deployment on Render**, so the model is reachable without any local setup

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.13 |
| Data & ML | pandas, NumPy, scikit-learn |
| Visualization (EDA) | Matplotlib, Seaborn |
| Experiment tracking | MLflow, DagsHub |
| Serving | Flask, Gunicorn |
| Environment & dependencies | uv, pyproject.toml |
| Testing | pytest |
| CI/CD | GitHub Actions |
| Deployment | Render |

## Project Structure

```
End-To-End-Heart-Disease-Prediction-MLOps/
├── .github/
│   └── workflows/              # CI/CD pipeline definitions
├── artifacts/                  # Generated pipeline outputs (data splits, trained models)
├── notebooks/                  # Exploratory data analysis and experimentation
├── src/
│   └── heart_disease/          # Core package: ingestion, preprocessing, training, prediction pipeline
├── static/                     # Static assets for the Flask app
├── templates/                  # HTML templates (index, result, error pages)
├── tests/                      # Test suite
├── app.py                      # Flask application entry point
├── testPredictions.py          # Manual/scripted prediction checks
├── template.py                 # Project scaffolding script
├── pyproject.toml              # Project metadata and dependencies
├── requirements.txt            # Pinned dependencies (for pip-based installs)
├── uv.lock                     # Locked dependency versions for uv
└── LICENSE
```

## Model Input Features

The prediction pipeline accepts standard clinical attributes associated with heart disease risk, such as:

| Feature | Description |
|---|---|
| `age` | Patient age in years |
| `sex` | Patient sex, encoded numerically |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar |
| `restecg` | Resting electrocardiographic results |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy |
| `thal` | Thalassemia-related categorical indicator |

The service returns a prediction (presence or absence of heart disease) based on the submitted values.

## Getting Started

### Prerequisites

- Python 3.10+ (see `.python-version`)
- Git
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
git clone https://github.com/RiteshSnippet/End-To-End-Heart-Disease-Prediction-MLOps.git
cd End-To-End-Heart-Disease-Prediction-MLOps
```

Using `uv` (recommended, uses the locked versions in `uv.lock`):

```bash
uv sync
```

Or using plain pip:

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Web Application

```bash
python app.py
```

The app runs locally at `http://localhost:5000` with a form to enter patient details and receive a heart disease risk prediction.

### Running Tests

```bash
pytest
```

You can also run `testPredictions.py` directly to sanity-check the prediction pipeline against sample inputs.

## CI/CD

GitHub Actions workflows under `.github/workflows/` run automated checks on push and pull requests, keeping the codebase tested before changes reach `main`.

## Deployment

The app is deployed on **[Render](https://render.com)** and served live at:

👉 **https://heart-disease-prediction-1lrm.onrender.com**

To deploy your own copy on Render:

1. Fork/clone this repository and push it to your own GitHub account.
2. Create a new **Web Service** on Render and connect it to your repo.
3. Set the build command (e.g. `pip install -r requirements.txt`) and start command (e.g. `python app.py` or a Gunicorn command if configured).
4. Deploy — Render will build and host the Flask app automatically on every push.

## Roadmap

- Add data versioning (DVC) and experiment tracking (MLflow) for full pipeline reproducibility
- Model monitoring and drift detection in production
- REST API endpoint alongside the existing web UI
- Model explainability (e.g., SHAP) for individual predictions
- Dockerized deployment alongside the Render hosting

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

**Ritesh Kumar Behera**
GitHub: [@RiteshSnippet](https://github.com/RiteshSnippet)

## Contributing

Issues and pull requests are welcome. For significant changes, open an issue first to discuss what you would like to change.