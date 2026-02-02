# Logistic Regression – Airline Passenger Satisfaction

Predicts airline passenger satisfaction using a Logistic Regression model with feature engineering for improved performance.

---

## Overview
This project applies **Logistic Regression** to classify airline passenger satisfaction based on survey data.  
The goal is to practice **data preprocessing**, **feature engineering** (including polynomial features), and **model evaluation** using a simple and interpretable model.

---

## Dataset
The project uses the **Airline Passenger Satisfaction dataset**, which is publicly available.  

- It is located in the `data/` folder.  

---

## Approach
- Load and clean the dataset (handle missing values, reduce noises)  
- Encode categorical features and scale numerical features  
- Generate **polynomial features** to capture non-linear relationships  
- Train a **Logistic Regression** model with L2 regularization  
- Evaluate performance using **accuracy** and **confusion matrix**  

> Note: Using polynomial features improved the model metrics while keeping it interpretable.

---

## Results
- Model accuracy: ~94%
- Polynomial features allowed the model to capture non-linear patterns and slightly improve performance
- Confusion matrix shows balanced classification of satisfied vs unsatisfied passengers

---

## Tools
- Python 3.x  
- Pandas, NumPy
- matplotlib
- Scikit-learn  
- Jupyter Notebook  

---

## How to Run
1. Clone the repository:
```bash
git clone https://github.com/E1-M0ra/logistic-regression-airline-satisfaction.git
