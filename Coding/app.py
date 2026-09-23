from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

MODEL_PATH = 'Models/hdi_model.pkl'
ENCODER_PATH = 'Models/label_encoder.pkl'

def get_model_and_encoder():
    """Loads model and encoder, or trains them dynamically with a fallback if CSV is empty."""
    os.makedirs('Models', exist_ok=True)
    
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
            model = joblib.load(MODEL_PATH)
            label_encoder = joblib.load(ENCODER_PATH)
            return model, label_encoder
    except Exception:
        pass 
    
    # Try reading hdi_dataset.csv
    try:
        df = pd.read_csv('hdi_dataset.csv')
        df.columns = df.columns.str.strip().str.lower()
    except Exception:
        df = pd.DataFrame()
        
    # Fallback dataset if CSV is empty or missing columns
    if df.empty or 'hdi' not in df.columns or len(df.dropna()) == 0:
        fallback_data = {
            'country': ['India', 'United States', 'Germany', 'Japan', 'Brazil', 'China', 'United Kingdom', 'Canada', 'Australia', 'India'],
            'year': [2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2021],
            'hdi': [0.645, 0.926, 0.947, 0.925, 0.754, 0.768, 0.932, 0.929, 0.951, 0.650]
        }
        df = pd.DataFrame(fallback_data)
    
    target_col = 'hdi'
    label_encoder = LabelEncoder()
    
    if 'country' in df.columns:
        df['country'] = df['country'].fillna('Unknown')
        df['country_encoded'] = label_encoder.fit_transform(df['country'].astype(str))
        feature_cols = [col for col in ['country_encoded', 'year'] if col in df.columns]
    else:
        feature_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col, errors='ignore').tolist()

    df = df.dropna(subset=[target_col])
    df[feature_cols] = df[feature_cols].fillna(0)
    
    X = df[feature_cols]
    y = df[target_col]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)
    return model, label_encoder

model, label_encoder = get_model_and_encoder()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        country_name = request.form['country'].strip()
        year = float(request.form['year'])
        
        try:
            country_encoded = label_encoder.transform([country_name])[0]
        except ValueError:
            return render_template('index.html', prediction_text=f"Error: Country '{country_name}' is not recognized by the model.")
        features = np.array([[country_encoded, year]])
        prediction = model.predict(features)[0]
        return render_template('index.html', prediction_text=f'Predicted HDI for {country_name} in {int(year)}: {prediction:.4f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error in processing input: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)