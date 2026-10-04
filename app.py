from flask import Flask, request, jsonify, render_template
import pickle
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

model_load_error = None
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
except Exception as e:
    print(f"Error loading model or metadata: {e}")
    model_load_error = f"{type(e).__name__}: {str(e)}"
    model = None
    metadata = {}

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
