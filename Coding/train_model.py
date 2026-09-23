import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Ensure Models directory exists
os.makedirs('Models', exist_ok=True)

print("Loading dataset...")
df = pd.read_csv('hdi_dataset.csv')
print("Dataset columns found:", df.columns.tolist())

# Clean column names (strip whitespace and lowercase)
df.columns = df.columns.str.strip().str.lower()

# Identify target column
target_col = 'hdi'
if target_col not in df.columns:
    target_col = df.columns[-1]

print(f"Target column selected: {target_col}")

# Drop rows ONLY if the target value itself is missing
df = df.dropna(subset=[target_col])

# Encode categorical columns like 'country' if present
label_encoder = LabelEncoder()
if 'country' in df.columns:
    df['country'] = df['country'].fillna('Unknown')
    df['country_encoded'] = label_encoder.fit_transform(df['country'].astype(str))
    feature_cols = [col for col in ['country_encoded', 'year'] if col in df.columns]
else:
    feature_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col, errors='ignore').tolist()

print(f"Features used for training: {feature_cols}")

# Fill any remaining missing values in features with 0 or median instead of dropping rows
df[feature_cols] = df[feature_cols].fillna(0)

X = df[feature_cols]
y = df[target_col]

print(f"Total usable samples for training: {len(X)}")

if len(X) == 0:
    raise ValueError("Error: The dataset has 0 samples after cleaning. Please check your 'hdi_dataset.csv' content.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
print("Training model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the working model and label encoder
joblib.dump(model, 'Models/hdi_model.pkl')
joblib.dump(label_encoder, 'Models/label_encoder.pkl')

print("Success! Fresh model and label encoder successfully trained and saved to 'Models/' directory.")