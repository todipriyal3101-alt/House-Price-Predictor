from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and data
model = pickle.load(open('RidgeModel.pkl', 'rb'))
data = pd.read_csv('Cleaned Data.csv')

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk      = int(request.form.get('bhk'))
    bath     = int(request.form.get('bath'))
    sqft     = float(request.form.get('total_sqft'))

    input_data = pd.DataFrame([[location, sqft, bath, bhk]],
                  columns=['location', 'total_sqft', 'bath', 'bhk'])

    prediction = model.predict(input_data)[0]

    return render_template('index.html',
                           prediction=round(prediction, 2),
                           locations=sorted(data['location'].unique()))

if __name__ == "__main__":
    app.run(debug=True, port=5001)