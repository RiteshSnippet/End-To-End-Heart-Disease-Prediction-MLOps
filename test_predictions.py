import pandas as pd
from src.heart_disease.pipeline.prediction_pipeline import PredictPipeline, CustomData

df = pd.read_csv("notebooks/Data/heart.csv")

sample = df.head(10)

pipeline = PredictPipeline()

print("\n" + "=" * 80)
print("REAL DATA PREDICTION TEST")
print("=" * 80)

for index, row in sample.iterrows():

    data = CustomData(
        age=int(row["age"]),
        sex=int(row["sex"]),
        cp=int(row["cp"]),
        trestbps=int(row["trestbps"]),
        chol=int(row["chol"]),
        fbs=int(row["fbs"]),
        restecg=int(row["restecg"]),
        thalach=int(row["thalach"]),
        exang=int(row["exang"]),
        oldpeak=float(row["oldpeak"]),
        slope=int(row["slope"]),
        ca=int(row["ca"]),
        thal=int(row["thal"])
    )

    input_df = data.get_data_as_dataframe()

    prediction, probability = pipeline.predict(input_df)

    actual = int(row["target"])
    predicted = int(prediction[0])

    status = "✓ CORRECT" if actual == predicted else "✗ WRONG"

    print(f"\nRow: {index}")
    print(f"Actual Target     : {actual}")
    print(f"Model Prediction  : {predicted}")

    if probability is not None:
        print(f"Probability       : {probability[0] * 100:.2f}%")

    print(f"Result            : {status}")

print("\n" + "=" * 80)