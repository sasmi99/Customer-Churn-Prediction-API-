# Telecom Customer Churn Prediction

**Project Overview**  
This project predicts whether a telecom customer is likely to churn (leave the service) using machine learning. It includes **data preprocessing, feature engineering, model training, evaluation**, and deployment via a **FastAPI API** for real-time predictions.

**Dataset**  
- [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- 7,043 customer records, 20 features including demographic, service usage, billing, and contract information.  
- Target variable: `Churn` (1 = churn, 0 = no churn).

---

## Features
Key features include:  
- `tenure` → number of months the customer stayed  
- `Contract` → monthly, one-year, or two-year plan  
- `MonthlyCharges` / `TotalCharges` → billing info  
- `InternetService` → DSL or Fiber optic  

---

## Preprocessing
- Cleaned missing values (e.g., `TotalCharges`)  
- Engineered features: `avg_monthly_charge`, `support_risk`  
- Encoded categorical variables using one-hot encoding  
- Scaled numerical features with `StandardScaler`  

---

## Models Trained
- **Logistic Regression**  
- **Random Forest Classifier**

**Evaluation Metrics**  

| Metric        | Logistic Regression | Random Forest |
|---------------|------------------|---------------|
| Accuracy      | 0.81             | 0.80          |
| Precision (Churn) | 0.67          | 0.65          |
| Recall (Churn)    | 0.54          | 0.53          |
| F1-score (Churn)  | 0.60          | 0.58          |

**Confusion Matrix**  
- Shows true positives, true negatives, false positives, and false negatives for churn prediction.  

**Final Model Selection**  
- Logistic Regression was selected as the final model because it achieved a **higher F1-score for the churn class**, which is critical for identifying customers at risk of leaving.

---

## Recommendation & Ranking Logic
- Model outputs **churn probability scores** for each customer.  
- Customers are **ranked by probability**, allowing targeted retention strategies.  
- Similarity logic: Customers with similar service usage, tenure, or billing patterns as churned customers can be flagged as higher risk.  

---

## Scaling & Production Considerations
- **Scaling to 100k+ records**: Use batch processing and vectorized operations; deploy multiple API instances behind a load balancer.  
- **Retraining**: Periodically retrain with new labeled data; update the model, scaler, and features.  
- **Monitoring**: Track API latency, error rates, model prediction distribution, and drift over time. Use logging and dashboards (e.g., Grafana, CloudWatch).  
- **Cloud Deployment Costs**: Consider compute, storage, and networking; serverless deployments or autoscaling clusters reduce costs.  

---

## Deployment
- Deployed via **FastAPI** for real-time predictions.  
- `/predict` endpoint accepts customer data in JSON format and returns:  
  ```json
  {
    "churn_prediction": 1,
    "churn_probability": 0.78
  }

