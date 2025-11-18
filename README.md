# 🚗 **Auto MPG Prediction – End-to-End ML Project with Streamlit Deployment**

## 📌 **Overview**

This project is a complete **end-to-end Machine Learning pipeline** built using the **UCI/Kaggle Auto MPG dataset**. It covers **EDA, data cleaning, visualization, feature engineering, model training, model evaluation, saving the best model, cloud deployment, and a fully interactive Streamlit web application**.

The final deployed application predicts a car's **Miles Per Gallon (MPG)** based on three features:

* **Acceleration**
* **Horsepower**
* **Weight**

This repository demonstrates your full ML workflow—from dataset to deployment.

---

## 🧩 **Problem Statement**

Fuel efficiency is a critical factor in automotive engineering. Predicting MPG helps:

* Understand vehicle performance
* Evaluate engine efficiency
* Assist manufacturers and consumers in decision-making

Given a car’s acceleration, horsepower, and weight, the task is to **predict its MPG using regression models**.

---

## 🎯 **Objective**

* Perform complete Exploratory Data Analysis (EDA)
* Clean and preprocess the dataset
* Visualize key relationships and distributions
* Train multiple regression models and select the best one
* Save the final model (`Auto-mpg_best_model.pkl`)
* Deploy the model using **Streamlit**
* Host the solution on the cloud with an interactive UI

---

## 📂 **Dataset**

**Source:** UCI / Kaggle
Dataset Link: [https://www.kaggle.com/datasets/uciml/autompg-dataset](https://www.kaggle.com/datasets/uciml/autompg-dataset)

**Features Used:**

* `acceleration`
* `horsepower`
* `weight`

**Target Variable:**

* `mpg`

---

## 🛠 **Tools & Libraries**

* **Python 3.x**
* **Pandas**
* **NumPy**
* **Matplotlib / Seaborn**
* **Scikit-Learn**
* **XGBoost**
* **Streamlit**
* **Joblib**
* **Requests**

---

## 🔧 **Model Architecture**

Multiple models were trained and compared:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* XGBRFRegressor

The best performing model was saved as:

```
Auto-mpg_best_model.pkl
```

---

## 🧼 **Data Preprocessing**

* Missing value handling
* Type conversions
* Outlier detection
* Feature selection
* Normalization / Scaling (if required)
* Cleaning rows with invalid `horsepower` entries

A clean, final dataframe `df_cleaned` was used for model training.

---

## 📊 **EDA & Visualization Summary**

Key visualizations included:

* Distribution plots of `mpg`, `horsepower`, `acceleration`, `weight`
* Correlation heatmap
* Pairplot to understand relationships
* Boxplots to detect outliers
* Regression lines for selected features

These insights guided model selection and feature importance.

---

## 📈 **Evaluation Metrics**

Models were evaluated using:

* **R² Score**
* **Mean Squared Error (MSE)**
* **RMSE**

The best model balanced accuracy and interpretability.

---

## 🌐 **Streamlit App**

A complete **interactive prediction UI** built using Streamlit.

### 🔗 Live App (Cloud Deployment)

*(Add the link once deployed)*

### 🧱 App Features

* Clean sidebar with project information
* Slider-based inputs: acceleration, horsepower, weight
* Real-time MPG prediction
* Batch CSV prediction + download option
* Fetches model from local file or GitHub raw URL

### 📁 File: `app.py`

Contains the full Streamlit application logic.

---

## 📄 **Project Structure**

```
📦 auto-mpg-end-to-end-ml-With-Streamlit-Deployment
│
├── app.py
├── requirements.txt
├── Auto-mpg_best_model.pkl
├── README.md
└── notebooks/ (optional EDA + training)
```

---

## 📦 **requirements.txt**

```
streamlit==1.25.0
pandas==2.1.2
numpy==1.26.2
scikit-learn==1.3.2
xgboost==2.1.1
joblib==1.3.2
requests==2.31.0
```

---

## 🧑‍💻 **Author**

**Pranav Gaikwad**
📧 Email: [gaikwadpranav988@gmail.com](mailto:gaikwadpranav988@gmail.com)
🔗 LinkedIn: [https://www.linkedin.com/in/pranav-gaikwad-0b94032a](https://www.linkedin.com/in/pranav-gaikwad-0b94032a)
💻 GitHub: [https://github.com/pranavgaikwad51](https://github.com/pranavgaikwad51)

---

## 🙏 **Acknowledgements**

* UCI Machine Learning Repository
* Kaggle Dataset Provider
* Streamlit Team
* Scikit-Learn Community

---

## 📜 **License**

This project is released under the **MIT License**.

You are free to use, modify, and distribute this project with attribution.

---

## ⭐ **If you like this project, consider giving the repository a star!**
