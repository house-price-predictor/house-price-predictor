import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def main():
    # Load data
    df = pd.read_csv('data/nepal_house_data.csv')

    # Feature selection
    features = ['location', 'land_area_sqft', 'floors', 'bedrooms', 'bathrooms', 'rcc_structure', 'plumbing', 'electricity']
    target = 'total_price'

    X = df[features]
    y = df[target]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing
    categorical_cols = ['location']
    numeric_cols = ['land_area_sqft', 'floors', 'bedrooms', 'bathrooms', 'rcc_structure', 'plumbing', 'electricity']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    # Model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=200, random_state=42))
    ])

    # Train
    print("Training model...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.4f}")

    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.pkl')

    # Save metadata
    metadata = {
        "model_name": "Gradient Boosting Regressor",
        "dataset_version": "v1.0 (Limbu-Unish/Real_State_Prediction)",
        "features": features,
        "evaluation_metrics": {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4)
        },
        "training_record_count": len(X_train)
    }

    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("Model and metadata saved to models/")

if __name__ == "__main__":
    main()
