from flask import Flask, render_template, request
from src.heart_disease.pipeline.prediction_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict_datapoint():

    prediction = None
    probability = None
    result_title = None
    result_message = None
    result_class = None

    if request.method == "POST":

        try:

            data = CustomData(
                age=int(request.form["age"]),
                sex=int(request.form["sex"]),
                cp=int(request.form["cp"]),
                trestbps=int(request.form["trestbps"]),
                chol=int(request.form["chol"]),
                fbs=int(request.form["fbs"]),
                restecg=int(request.form["restecg"]),
                thalach=int(request.form["thalach"]),
                exang=int(request.form["exang"]),
                oldpeak=float(request.form["oldpeak"]),
                slope=int(request.form["slope"]),
                ca=int(request.form["ca"]),
                thal=int(request.form["thal"])
            )

            final_data = data.get_data_as_dataframe()

            predict_pipeline = PredictPipeline()

            prediction, prediction_probability = (
                predict_pipeline.predict(final_data)
            )

            print("\n" + "=" * 60)
            print("FLASK DEBUG")
            print("=" * 60)

            print("Input Data:")
            print(final_data)

            print("Raw Prediction:")
            print(prediction)

            print("Raw Probability:")
            print(prediction_probability)

            print("=" * 60)

            prediction = int(prediction[0])

            if prediction_probability is not None:
                probability = round(
                    float(prediction_probability[0]) * 100, 2
                )

            if prediction == 1:

                result_title = "Heart Disease Detected"

                result_message = (
                    "The model predicts a positive result "
                    "based on the information provided."
                )

                result_class = "danger"

            else:

                result_title = "No Heart Disease Detected"

                result_message = (
                    "The model predicts a negative result "
                    "based on the information provided."
                )

                result_class = "success"

        except Exception as e:

            result_title = "Prediction Error"

            result_message = (
                "Something went wrong while processing "
                "the prediction. Please check your inputs "
                "and try again."
            )

            result_class = "error"

            print(f"Prediction error: {e}")

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        result_title=result_title,
        result_message=result_message,
        result_class=result_class
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)