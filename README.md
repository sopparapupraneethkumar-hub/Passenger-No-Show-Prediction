# Passenger No-Show Prediction Using Apache Spark (PySpark)

## Project Overview

Passenger No-Show Prediction is a data analytics and machine learning project that uses Apache Spark (PySpark MLlib) to analyze booking patterns and predict the likelihood of booking cancellations or no-shows.

The project implements a complete data processing and machine learning pipeline, starting from raw hotel booking data through data cleaning, exploratory data analysis (EDA), feature engineering, model training, prediction, and performance evaluation.

An interactive Streamlit dashboard is developed to visualize booking trends, identify high-risk bookings, compare actual and predicted no-show rates, and evaluate model performance.

The Hotel Booking Demand dataset is used as a stand-in for passenger booking data. The target variable represents whether a booking was cancelled.

## Objectives

- Process and analyze large-scale booking data using Apache Spark.
- Perform data cleaning and handle missing and duplicate records.
- Identify booking patterns and factors associated with cancellations.
- Apply feature engineering and categorical encoding.
- Build predictive classification models using Spark MLlib.
- Compare standard and class-weighted Logistic Regression.
- Evaluate model performance using multiple classification metrics.
- Develop an interactive dashboard for business-oriented insights.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Big Data Processing | Apache Spark, PySpark |
| Machine Learning | Spark MLlib |
| Algorithm | Logistic Regression |
| Data Manipulation | Pandas, NumPy |
| Visualization | Altair |
| Dashboard | Streamlit |
| Data Format | CSV |
| Version Control | Git, GitHub |

## Project Workflow

### 1. Data Loading and Exploration
- Load the booking dataset using PySpark.
- Examine the dataset schema, dimensions, and data types.
- Analyze missing values, duplicate records, and invalid bookings.
- Understand the distribution of booking cancellations.

### 2. Data Cleaning and Preprocessing
- Remove duplicate records.
- Handle missing values in important categorical and numerical columns.
- Remove invalid bookings with zero nights or no guests.
- Apply logical filtering to improve data quality.
- Remove columns that could cause target leakage.

### 3. Exploratory Data Analysis (EDA)
- Analyze booking cancellation rates across different customer segments.
- Examine booking lead times and booking windows.
- Study monthly booking trends.
- Compare cancellation patterns across hotel types, customer types, and market segments.

### 4. Feature Engineering
- Create total nights and booking window features.
- Generate agent and company availability indicators.
- Apply StringIndexer for categorical encoding.
- Apply OneHotEncoder to transform categorical features.
- Use VectorAssembler to combine features.
- Apply StandardScaler to numerical features.

### 5. Machine Learning Model Development
- Split the dataset into 80% training and 20% testing data.
- Develop a Logistic Regression classification pipeline using Spark MLlib.
- Train a standard Logistic Regression model.
- Implement a class-weighted Logistic Regression model to address class imbalance.
- Generate predictions and no-show probabilities.

### 6. Model Evaluation
Evaluate the models using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Baseline Accuracy

Compare standard and class-weighted Logistic Regression to understand the trade-off between detecting more no-shows and generating false alarms.

### 7. Interactive Streamlit Dashboard

The dashboard provides seven interactive sections:

**Overview**
- Total cleaned bookings
- Overall no-show rate
- Model accuracy and ROC-AUC
- Data cleaning statistics

**Segment Analysis**
- Cancellation rates across booking segments
- Hotel and customer-type comparisons
- Booking window analysis

**Predicted vs Actual**
- Actual versus predicted cancellation percentages
- Segment-level comparisons
- Standard and class-weighted model selection

**Model Performance**
- Dynamic decision threshold
- Confusion matrix
- ROC curve
- Precision and recall analysis

**Feature Importance and Risk Drivers**
- Logistic Regression coefficients
- Positive and negative feature relationships
- Odds ratio analysis

**Booking Trends**
- Monthly booking volume
- Monthly cancellation trends

**Explore Bookings**
- Low, medium, and high-risk booking categories
- Highest-risk booking identification
- Downloadable prediction results

## Generated Outputs

The PySpark pipeline generates the following analysis-ready CSV files:

| File | Description |
|---|---|
| predictions.csv | Actual labels, predicted labels, and no-show probabilities |
| metrics.csv | Dataset statistics and model evaluation metrics |
| model_comparison.csv | Comparison of standard and class-weighted models |
| confusion_matrix.csv | Actual versus predicted classification counts |
| feature_coefficients.csv | Logistic Regression coefficients and odds ratios |
| eda_segments.csv | Segment-level booking and cancellation analysis |
| eda_monthly.csv | Monthly booking trends and cancellation percentages |

## Project Architecture

Raw Booking Dataset → PySpark Data Processing → Data Cleaning → Feature Engineering → Spark MLlib Pipeline → Logistic Regression → Model Evaluation → CSV Output Generation → Streamlit Dashboard

## Deployment

The interactive dashboard is deployed using Streamlit Community Cloud.

**Live Application:**  
https://passenger-no-show-prediction-itgeby7kgjbq8krefabwnv.streamlit.app/

## Key Learning Outcomes

- Practical implementation of distributed data processing using PySpark.
- Building end-to-end machine learning pipelines with Spark MLlib.
- Understanding feature engineering and categorical encoding.
- Handling class imbalance through class weighting.
- Evaluating classification models beyond accuracy.
- Developing interactive dashboards for analytical decision-making.
- Deploying Python applications using GitHub and Streamlit Community Cloud.

## Future Scope

- Explore advanced classification algorithms such as Random Forest and Gradient Boosting.
- Implement hyperparameter tuning and cross-validation.
- Integrate real-time booking data for dynamic prediction.
- Develop an API-based prediction service.
- Improve model explainability using SHAP.
- Extend the application to real passenger transportation datasets.

## Repository Structure

```text
Passenger-No-Show-Prediction/
│
├── dashboard.py
├── Project16_PySpark_final.py
├── requirements.txt
│
└── project16_outputs/
    ├── predictions.csv
    ├── metrics.csv
    ├── model_comparison.csv
    ├── confusion_matrix.csv
    ├── feature_coefficients.csv
    ├── eda_segments.csv
    ├── eda_monthly.csv
    └── predicted_vs_actual.png
```

## Author

**Praneeth Kumar Sopparapu**

B.Tech – Computer Science and Engineering

GitHub: https://github.com/sopparapupraneethkumar-hub

---

*This project demonstrates an end-to-end big data analytics and machine learning workflow using Apache Spark, PySpark MLlib, Python, and Streamlit.*
