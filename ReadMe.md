# ***ProdGuard: Product Lifecycle \& Decline Risk Monitoring***



### ***Overview***

ProdGuard is a Python-based analytics solution designed to identify Product Life Cycle (PLC) stages and predict early decline risk at the SKU level using historical sales and customer data. The project enables proactive business decisions by detecting early warning signals of product decline before significant revenue loss occurs.



### ***Problem Statement***

* Traditional reporting methods detect product decline only after major revenue impact. Organizations lack:
* Early warning detection
* Scalable SKU-level monitoring
* Predictive insights for lifecycle transitions
* Data-driven intervention strategies
* ProdGuard addresses these gaps using feature engineering, rule-based logic, and machine learning.



### ***Solution Approach***

##### <i>1. Data Processing</i>

* Transaction-level sales, pricing, promotion, and customer data
* Monthly SKU-level aggregation
* Missing value handling and outlier treatment
* Strict prevention of data leakage

##### *2. Feature Engineering*

Key features include:

* Sales growth rate
* Rolling averages (3M, 6M)
* Drop from peak (%)
* Time since peak
* Consecutive negative growth
* Sales volatility \& momentum
* Product age (months)

##### *3. PLC Stage Classification*

Rule-based logic classifies products into:

* Introduction
* Growth
* Maturity
* Decline

##### *4. Decline Risk Prediction*

Supervised machine learning models:

* Logistic Regression (baseline)
* Random Forest (final model)

Random Forest was selected due to superior recall and better handling of non-linear patterns.



#### ***Model Evaluation***

Evaluation metrics:

* Accuracy
* Precision
* Recall (business-priority metric)
* F1-score

The model predicts 3-month forward decline risk and categorizes SKUs into:

* Low Risk
* Medium Risk
* High Risk

Each high-risk SKU includes explanation and recommended action.



#### ***Additional Analytics***

* Region-wise high-risk SKU identification
* Market Basket Analysis (bundle recommendations)
* Portfolio-level risk monitoring



### ***Tech Stack***



* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* MLxtend



### ***Business Impact***



* Early decline detection
* Reduced revenue risk
* Targeted interventions
* Data-driven SKU rationalization
* Portfolio-level monitoring



###### ***Author***



***Aditi***

***PG Diploma in Big Data Analytics***

***C-DAC Noida***


