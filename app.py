from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the saved model
model = joblib.load('churn_model.pkl')  # Update with the correct model file name

@app.route('/predict_churn', methods=['POST'])
def predict_churn():
    data = request.get_json()
    df = pd.DataFrame(data)
    predictions = model.predict(df)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
