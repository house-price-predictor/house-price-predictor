# Nepal Property Dataset

## Overview
This dataset contains real estate listings from Nepal, providing a structured source of property features and their corresponding market prices.

## Source Details
- **Dataset Name:** Nepal House Price Dataset
- **Source:** Originally derived from GitHub repository [Limbu-Unish/Real_State_Prediction](https://github.com/Limbu-Unish/Real_State_Prediction). The original maintainers collected and cleaned this data from Nepal real estate listings.
- **Geographic Coverage:** Major property markets in Nepal, including Kathmandu, Lalitpur, Pokhara, Chitwan, Dhangadi, and Syangja.
- **Number of Records:** 1000 properties

## Dataset Features
The dataset contains the following attributes used in this valuation system:
1. `location`: The city or district where the property is located.
2. `land_area_sqft`: The total land area of the property, measured in square feet (which can be converted to local units like Aana or Dhur).
3. `floors`: Number of floors in the property.
4. `bedrooms`: Number of bedrooms.
5. `bathrooms`: Number of bathrooms.
6. `rcc_structure`: Binary indicator (1 = Yes, 0 = No) showing whether the property is built with an RCC (Reinforced Cement Concrete) pillar structure.
7. `plumbing`: Binary indicator for plumbing availability.
8. `electricity`: Binary indicator for electricity access.
9. `total_price` (Target Variable): The total valuation or listing price of the property in Nepalese Rupees (NPR/Rs).

*(Note: Attributes like `windows`, `doors`, `cement_bags`, and granular construction costs were omitted from the machine learning pipeline to avoid data leakage and maintain an accessible user experience).*

## Preprocessing
- **Missing values:** Handled via imputation where appropriate.
- **Categorical encoding:** One-Hot Encoding for the `location` variable.
- **Scaling:** `StandardScaler` applied to numeric variables (area, floors, bedrooms, bathrooms) before training.

## Limitations & Known Biases
- **Sample Size:** The model is trained on ~800 records (after train/test split). This is a relatively small dataset and might not capture extreme outliers or hyper-localized pricing trends accurately.
- **Temporal Bias:** The date of listing extraction is not strictly versioned, so predictions represent the market at the time of data collection and may not perfectly reflect real-time inflation or market crashes.
- **Informational Use:** The output is a statistical estimate for academic demonstration and not a certified professional valuation.
