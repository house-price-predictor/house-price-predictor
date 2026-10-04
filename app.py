from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import json
import traceback

import os

app = Flask(__name__)

# Load model and metadata at startup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "models", "metadata.json")

def train_model():
    """Train and return a fresh model using whatever sklearn version is installed."""
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    DATA_PATH = os.path.join(BASE_DIR, "data", "nepal_house_data.csv")
    df = pd.read_csv(DATA_PATH)

    features = ['location', 'land_area_sqft', 'floors', 'bedrooms', 'bathrooms', 'rcc_structure', 'plumbing', 'electricity']
    target = 'total_price'

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), ['land_area_sqft', 'floors', 'bedrooms', 'bathrooms', 'rcc_structure', 'plumbing', 'electricity']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['location'])
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=200, random_state=42))
    ])

    print("Training fresh model on server...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    meta = {
        "model_name": "Gradient Boosting Regressor",
        "features": features,
        "evaluation_metrics": {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)},
        "training_record_count": len(X_train)
    }

    # Save fresh model for next startup
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=4)

    print(f"Fresh model trained: R2={r2:.4f}")
    return pipeline, meta

model_load_error = None
model = None
metadata = {}

# Try loading saved model first
try:
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    print("Model loaded from disk successfully.")
except Exception as e:
    print(f"Could not load saved model ({e}). Training fresh model...")
    model_load_error = f"{type(e).__name__}: {str(e)}"
    try:
        model, metadata = train_model()
        model_load_error = None
    except Exception as e2:
        print(f"Fresh training also failed: {e2}")
        model_load_error = f"Training failed: {str(e2)}"

@app.route('/')
def home():
    return render_template('index.html', metadata=metadata)

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

def format_currency_npr(amount):
    """
    Formats the amount to Nepalese numbering system (Lakhs, Crores).
    """
    try:
        amount = float(amount)
        if amount < 0:
            return "Invalid Price"
        # Simplistic format: convert to string and format with commas (international)
        # To do Nepali format (xx,xx,xxx)
        s = str(int(amount))
        if len(s) <= 3:
            formatted = s
        else:
            last3 = s[-3:]
            other = s[:-3]
            # split 'other' into chunks of 2
            chunks = [other[max(i-2, 0):i] for i in range(len(other), 0, -2)]
            chunks.reverse()
            formatted = ",".join(chunks) + "," + last3
        
        # Add labels
        label = ""
        if amount >= 10000000:
            label = f" ({amount/10000000:.2f} Crore)"
        elif amount >= 100000:
            label = f" ({amount/100000:.2f} Lakh)"
            
        return f"Rs {formatted}{label}"
    except:
        return "N/A"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({"error": f"Model not loaded. Reason: {model_load_error}"}), 500

        # Whether it's an API request (JSON) or form submission
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        # Extract features
        location = data.get('location')
        land_area_sqft = float(data.get('land_area_sqft', 0))
        floors = int(data.get('floors', 1))
        bedrooms = int(data.get('bedrooms', 1))
        bathrooms = int(data.get('bathrooms', 1))
        rcc_structure = int(data.get('rcc_structure', 1))
        plumbing = int(data.get('plumbing', 1))
        electricity = int(data.get('electricity', 1))

        # Basic validation
        if not location:
            return jsonify({"error": "Location is required"}), 400
        if land_area_sqft <= 0:
            return jsonify({"error": "Land area must be greater than zero"}), 400
        if floors < 0 or bedrooms < 0 or bathrooms < 0:
            return jsonify({"error": "Floors, bedrooms, and bathrooms cannot be negative"}), 400

        # Prepare for prediction
        input_data = pd.DataFrame([{
            'location': location,
            'land_area_sqft': land_area_sqft,
            'floors': floors,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'rcc_structure': rcc_structure,
            'plumbing': plumbing,
            'electricity': electricity
        }])

        prediction = model.predict(input_data)[0]
        
        if prediction < 0:
            return jsonify({"error": "Model predicted a negative price, which is invalid."}), 400

        # Add a rough confidence interval (e.g. based on RMSE)
        rmse = metadata.get("evaluation_metrics", {}).get("RMSE", 414674)
        lower_bound = max(0, prediction - rmse)
        upper_bound = prediction + rmse

        price_per_sqft = prediction / land_area_sqft if land_area_sqft > 0 else 0

        # Construct response
        result = {
            "predicted_price_raw": float(prediction),
            "predicted_price_formatted": format_currency_npr(prediction),
            "range_lower_formatted": format_currency_npr(lower_bound),
            "range_upper_formatted": format_currency_npr(upper_bound),
            "price_per_sqft_formatted": format_currency_npr(price_per_sqft),
            "model_metadata": metadata,
            "input_echo": data
        }

        if request.is_json:
            return jsonify(result)
        else:
            return render_template('index.html', result=result, metadata=metadata)

    except ValueError as ve:
        return jsonify({"error": f"Invalid data type: {str(ve)}"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred"}), 500

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
