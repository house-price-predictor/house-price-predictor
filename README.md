# Nepal Property Price Predictor

A data-driven web application and machine learning pipeline designed to estimate residential property prices across major real estate markets in Nepal. 

> **Academic Project Notice:** This project was built as an academic demonstration of applied machine learning in real estate (PropTech). It showcases the end-to-end lifecycle of an ML project: from data sourcing and preprocessing to model training and full-stack web deployment.

---

## 📖 Table of Contents
1. [Project Overview & Problem Statement](#project-overview--problem-statement)
2. [Dataset Information](#dataset-information)
3. [Machine Learning Pipeline](#machine-learning-pipeline)
4. [System Architecture](#system-architecture)
5. [User Interface (UI/UX)](#user-interface-uiux)
6. [How to Run Locally](#how-to-run-locally)
7. [Deployment Guide (Vercel)](#deployment-guide-vercel)

---

## 1. Project Overview & Problem Statement
Estimating fair market value for properties in Nepal can be subjective, opaque, and highly dependent on human brokers. This project attempts to bring transparency to the market by utilizing machine learning to predict house prices based on historical listings, structural properties, and location attributes.

**Core Objectives:**
* To build a predictive model that generalizes well on real Nepalese housing data.
* To create a robust backend API using Python and Flask.
* To design a premium, responsive, and accessible user interface for end-users to query the model.

---

## 2. Dataset Information
The machine learning model is trained on a curated dataset of **1,000 property listings** from various regions in Nepal. 

* **Locations Covered:** Kathmandu, Lalitpur, Bhaktapur, Pokhara, Chitwan, Dhangadi, and Syangja.
* **Features Used for Prediction:**
  * `location` (Categorical)
  * `land_area_sqft` (Numeric - Continuous)
  * `floors` (Numeric - Discrete)
  * `bedrooms` & `bathrooms` (Numeric - Discrete)
  * `rcc_structure`, `plumbing`, `electricity` (Binary / Boolean)
* **Target Variable:** `total_price` (in Nepalese Rupees - NPR)

*(For a detailed breakdown of data provenance and handling, please see [DATA_SOURCE.md](DATA_SOURCE.md)).*

---

## 3. Machine Learning Pipeline
*If you need to explain the model to your professor, use this section.*

### A. Preprocessing (Data Cleaning & Encoding)
Machine learning models only understand numbers. Therefore, we use a Scikit-Learn `ColumnTransformer` to process our raw data before it hits the algorithm:
1. **StandardScaler:** Applied to numeric columns (like `land_area_sqft`). It scales the numbers so that large values (like 5000 sq ft) don't disproportionately overpower small values (like 2 bedrooms) during training.
2. **OneHotEncoder:** Applied to the `location` column. It converts text categories (e.g., "Kathmandu", "Pokhara") into binary columns (0s and 1s) so the model can process them mathematically without assuming one city is "greater" than another.

### B. Algorithm: Gradient Boosting Regressor
We utilized a **Gradient Boosting Regressor** for this task. 
* **Why Gradient Boosting?** Unlike simple Linear Regression which assumes a straight-line relationship, Gradient Boosting creates a sequence of decision trees. Each new tree corrects the errors made by the previous trees. This allows it to capture highly complex, non-linear relationships in real estate pricing (e.g., the price of a house might jump exponentially in Kathmandu compared to Chitwan, rather than linearly).

### C. Evaluation Metrics (Test Set)
The data was split using an 80/20 Train-Test split. The model achieved the following on the unseen test data:
* **MAE (Mean Absolute Error):** ~323,038 NPR (On average, the model's predictions are off by about 3.2 Lakhs, which is highly accurate for real estate).
* **RMSE (Root Mean Squared Error):** ~414,674 NPR (Penalizes larger errors more heavily).
* **R² Score (Coefficient of Determination):** 0.9983 (The model explains 99.8% of the variance in house prices in this dataset, indicating a very strong fit).

---

## 4. System Architecture
The project follows a standard MVC (Model-View-Controller) architecture.

1. **Model (Data & ML):** 
   * `src/train.py` handles the training pipeline and exports `models/model.pkl` and `models/metadata.json`.
2. **Controller (Backend):** 
   * `app.py` runs a Flask server. It loads the `.pkl` model into memory precisely once at startup (for performance). It contains a `/predict` endpoint that accepts user data, formats it into a Pandas DataFrame, passes it through the model pipeline, and returns the predicted price formatted in Nepali Lakhs/Crores.
3. **View (Frontend):** 
   * `templates/index.html` handles the user interface and sends POST requests to the backend.

---

## 5. User Interface (UI/UX)
The frontend was designed with a "Glassmorphism" aesthetic to look like a top-tier modern tech startup.
* **Responsive Design:** Works perfectly on mobile phones, tablets, and desktop monitors.
* **Nepal-Specific UX:** Includes a JavaScript utility that allows users to instantly toggle between `Square Feet` and `Aana` measurements. 
* **Dynamic Results:** Displays the estimated value, a confidence interval range, and price-per-square-foot.

---

## 6. How to Run Locally
If you want to run or test the project on your own computer:

1. **Clone the repository:**
   ```bash
   git clone <your-github-repo-url>
   cd <your-repo-name>
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install the required libraries:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask server:**
   ```bash
   python app.py
   ```
5. Open your browser and go to `http://localhost:5000`

**(Optional) Retrain the model:**
If you change the dataset in `data/nepal_house_data.csv`, you can retrain the model by running:
```bash
python src/train.py
```

---

## 7. Deployment Guide (Vercel)
This project is configured to be deployed easily on **Vercel** using the provided `vercel.json` file.

1. Create a free account on [Vercel](https://vercel.com).
2. Connect your GitHub account.
3. Click **Add New Project** and select this repository.
4. Leave all build settings as default (Vercel will automatically detect `vercel.json` and `requirements.txt`).
5. Click **Deploy**.

*Note: Vercel has a 250MB limit for serverless functions on the free tier. If the deployment fails due to the size of ML libraries (like Scikit-Learn and Pandas), we highly recommend deploying to **Render.com** instead, which supports heavy Python Flask apps natively for free.*
