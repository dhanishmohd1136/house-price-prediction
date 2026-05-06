# 🏠 Housing Price Prediction

##  Project Overview

This project aims to predict house prices using regression models on the Ames Housing dataset.
The workflow includes data preprocessing, feature engineering, model training, and evaluation.

---

##  Dataset

* Source: Ames Housing Dataset
* Samples: 1460 houses
* Features: 80+ (numerical + categorical)
* Target: `SalePrice`

---

##  Preprocessing Steps

* Log transformation applied to `SalePrice` to reduce skewness
* Missing values handled:

  * Categorical → filled with `"None"`
  * Numerical → filled with median
* Feature transformations:

  * `LotArea` log-transformed
  * `MSSubClass` converted to categorical
* One-hot encoding applied to categorical variables
* Dropped low-information features (`Street`, `Utilities`)

---

## Models Used

* Linear Regression
* Ridge Regression (L2 Regularization)
* Lasso Regression (L1 Regularization)
* Polynomial Regression (degree = 2)

---

##  Model Performance

| Model      | Train R² | Test R² | Train RMSE | Test RMSE |
| ---------- | -------- | ------- | ---------- | --------- |
| Linear     | 0.9479   | 0.8587  | 0.0891     | 0.1624    |
| Ridge      | 0.9354   | 0.8689  | 0.0992     | 0.1564    |
| Lasso      | 0.8956   | 0.8801  | 0.1262     | 0.1496    |
| Polynomial | 1.0000   | 0.8525  | ~0.0000    | 0.1659    |

---

##  Key Insights

* **Lasso Regression performed best on test data**

  * Highest Test R²: **0.8801**
  * Lowest Test RMSE: **0.1496**
  * Performs implicit feature selection → reduces overfitting

* **Polynomial Regression severely overfits**

  * Perfect training score (R² = 1.0)
  * Worse generalization → classic overfitting case

* **Ridge Regression provides stable performance**

  * Good balance between bias and variance

* **Linear Regression performs well but slightly worse than regularized models**

---

##  Overfitting Analysis

| Model      | Observation         |
| ---------- | ------------------- |
| Polynomial | Severe overfitting  |
| Linear     | Mild overfitting    |
| Ridge      | Controlled variance |
| Lasso      | Best generalization |

---

##  Conclusion

* Regularization is essential due to:

  * High dimensionality (many features after encoding)
  * Multicollinearity

* **Best model: Lasso Regression**

  * Provides optimal balance between accuracy and generalization

---

##  Future Improvements

* Feature engineering (TotalSF, HouseAge)
* Hyperparameter tuning (GridSearchCV)
* Cross-validation for more robust evaluation
* Advanced models (XGBoost, Random Forest)

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn

---

##  Project Structure

```
housing-price-prediction/
├── data/
├── notebooks/
├── src/
├── models/
├── reports/
```

---

##  How to Run

```bash
pip install -r requirements.txt
```

Run notebooks in order:

1. EDA
2. Feature Engineering
3. Modeling
4. Comparison

---

##  Final Result

```text
Best Model → Lasso Regression
Test R²     → 0.8801
Test RMSE   → 0.1496
```

---


