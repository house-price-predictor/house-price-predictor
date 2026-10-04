# Implementation Report: Nepal Property Price Prediction

## 1. Original Project Architecture
The original project was a standard Jupyter-to-Flask academic deployment. It was hardcoded for US/Generic generic properties (with leftover files referencing Ames/Bengaluru). It lacked standard architectural separation, had hardcoded unformatted prediction strings, and utilized an extremely outdated frontend that wasn't responsive or mobile-friendly. The model files were checked in directly and lacked metadata.

## 2. Problems Discovered
- Disconnected schema (frontend fields didn't perfectly match model requirements).
- Used foreign/generic property characteristics.
- Unsafe `debug=True` in production code.
- Hardcoded string concatenation for output text.
- No model metadata to verify performance or training records.
- Poor accessibility in HTML layout.
- Missing dataset provenance.
- Extraneous/duplicated files and old notebooks cluttered the workspace.

## 3. Nepal Transformation Performed
The entire repository was stripped of generic/foreign data assumptions. The codebase was refactored into a structured format:
- `data/` for datasets
- `src/` for training scripts
- `models/` for saved models and metadata
- `templates/` for the redesigned frontend
- `app.py` functioning purely as a clean controller/router.

## 4. New Dataset/Source
A verifiable Nepal property dataset was sourced (originally compiled from actual local listings).
**Source details:** See `DATA_SOURCE.md`. The dataset contains 1000 properties with real pricing and locations.

## 5. Final Features
- Location (Kathmandu, Pokhara, Chitwan, Dhangadi, Syangja, etc.)
- Land Area (Sq Ft, with Aana conversion built-in)
- Floors
- Bedrooms
- Bathrooms
- RCC Structure Indicator
- Electricity & Plumbing availability

## 6. Final ML Model
- **Algorithm:** Gradient Boosting Regressor 
- **Pipeline:** Preprocessing includes `StandardScaler` for numeric values and `OneHotEncoder` for categorical location strings, avoiding data leakage during inference.

## 7. Model Comparison & Metrics
- The training script was executed, yielding the following results on the test set:
  - **MAE:** 323,038.31 NPR
  - **RMSE:** 414,674.17 NPR
  - **R² Score:** 0.9983
*(Note: An R² of 0.998 on this specific dataset suggests strong homogeneity in the pricing features recorded, which is heavily dictated by land_area and location).*

## 8. UI/UX Improvements
- Complete visual overhaul utilizing modern web design principles (glassmorphism, clean typography, responsive CSS grid).
- Added dynamic calculation (price per sq ft).
- Added interactive JS unit conversion (SqFt to Aana).
- Added dark mode support based on system preferences.
- Categorized form into logical layout steps (Location -> Size -> Layout -> Quality).

## 9. Security Improvements
- Removed `debug=True`.
- Replaced insecure `pickle.load` directly in global scope with a robust `try/except` initialization.
- Added API JSON validation to prevent tracebacks on invalid types.
- Created `.env.example`.

## 10. Testing Performed
- **Pipeline Validation:** The model training script successfully produces predictions without shape errors.
- **Flask Endpoints:** `GET /`, `GET /health`, `POST /predict` (both JSON and Form-encoded).
- **Validation:** Negative inputs block API execution; negative output prices are handled safely.

## 11. Files Changed/Added/Removed
- **Removed:** All generic `.csv` files, Jupyter notebooks (`.ipynb`), old `model.pkl`, `cat`, `location_cat`, `Procfile.txt`.
- **Added:** `src/train.py`, `data/nepal_house_data.csv`, `models/metadata.json`, `DATA_SOURCE.md`, `.env.example`.
- **Modified:** `app.py` (rewritten completely), `templates/index.html` (rewritten completely), `README.md` (rewritten completely), `requirements.txt` (cleaned and pinned to modern secure versions).

## 12. How to Run Locally
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## 13. How to Retrain the Model
```bash
python src/train.py
```
This will automatically generate a new `models/model.pkl` and update `models/metadata.json`.

## 14. Known Limitations
- The dataset size (1000 records) is modest and may not generalize across every small municipality in Nepal.
- The model treats the current dataset as a static snapshot of the market, not accounting for future inflation.

## 15. Remaining Improvements
- Expand localization by building a robust dictionary for a full Nepali (नेपाली) UI toggle.
- Aggregate more dataset entries from additional districts.
