from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and features
model = joblib.load('trained_data/student_pass_fail_model.pkl')
features = ['Study_Hours_per_Week', 'Sleep_Hours_per_Night', 'Participation_Score', 'Attendance (%)']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Prepare input dataframe with exact feature order
    input_data = {key: data[key] for key in features}
    input_df = pd.DataFrame([input_data])

    # Predict pass/fail class
    pred_class = model.predict(input_df)[0]

    # Predict probabilities (sigmoid output)
    pred_proba = model.predict_proba(input_df)[0]
    prob_fail = pred_proba[0] * 100
    prob_pass = pred_proba[1] * 100

    response = {
        "prediction": "Pass" if pred_class == 1 else "Fail",
        "probability_pass": round(prob_pass, 2),
        "probability_fail": round(prob_fail, 2)
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
