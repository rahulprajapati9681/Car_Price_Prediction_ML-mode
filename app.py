from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('C:\\Users\\RAHUL\\OneDrive\\Desktop\\codes\\ML\\Car price prediction\\LinearRegressionModel.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')  # Ensure index.html is in a 'templates' folder

@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form['name']
        company = request.form['company']
        year = int(request.form['year'])
        kms_driven = int(request.form['kms_driven'])
        fuel_type = request.form['fuel_type']
        
        # Prepare input data for prediction (modify as per your model's requirement)
        input_data = pd.DataFrame([[name, company, year, kms_driven, fuel_type]],columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
        
        # Predict car price
        prediction = model.predict(input_data)
        
        return jsonify({'predicted_price': round(prediction[0], 2)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
