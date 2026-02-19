from flask import Flask, render_template, request
import numpy as np
import pickle
import os

# Ensure paths are resolved relative to this script so app can be
# started from any working directory.
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "random_forest_model.pkl")
scaler_path = os.path.join(base_dir, "scaler.pkl")

app = Flask(__name__)

# Load trained model and scaler
model = None
scaler = None
try:
    with open(model_path, "rb") as mf:
        model = pickle.load(mf)
    with open(scaler_path, "rb") as sf:
        scaler = pickle.load(sf)
except Exception as e:
    # Print the error to console for easier debugging when starting the app
    print(f"Error loading model or scaler: {e}")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = [
            float(request.form['age_first_funding_year']),
            float(request.form['age_last_funding_year']),
            float(request.form['age_first_milestone_year']),
            float(request.form['age_last_milestone_year']),
            float(request.form['relationships']),
            float(request.form['funding_rounds']),
            float(request.form['funding_total_usd']),
            float(request.form['milestones'])
        ]

        # Scale input
        scaled_input = scaler.transform([input_features])

        # Predict
        prediction = model.predict(scaled_input)

        result = "Startup is Likely to Succeed (Acquired)" if prediction[0] == 1 else "Startup is Likely to Fail (Closed)"

        return render_template('result.html', prediction_text=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Disable the auto-reloader to prevent double-import issues
    # when loading heavy compiled extensions like scipy.
    app.run(debug=True, use_reloader=False)
