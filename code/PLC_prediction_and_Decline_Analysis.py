# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 18:27:39 2026

@author: saxen
"""

import pandas as pd
import numpy as np

products=pd.read_csv('C:/Users/saxen/OneDrive/Desktop/Plc Prediction Dataset/fmcg_dirty.csv')
customers=pd.read_csv('C:/Users/saxen/OneDrive/Desktop/Plc Prediction Dataset/customers_dirty.csv')
transactions=pd.read_csv('C:/Users/saxen/OneDrive/Desktop/Plc Prediction Dataset/transactions_dirty.csv')

print(products.shape)
print(transactions.shape)
print(customers.shape)

products.head()
transactions.head()
customers.head()

products["date"] = pd.to_datetime(products["date"], errors="coerce", dayfirst=True)

transactions["transaction_date"] = pd.to_datetime(
    transactions["transaction_date"],
    errors="coerce",
    dayfirst=True
)

customers["customer_since"] = pd.to_datetime(
    customers["customer_since"],
    errors="coerce",
    dayfirst=True
)

products["date"].isna().sum()
transactions["transaction_date"].isna().sum()
customers["customer_since"].isna().sum()

products=products.dropna(subset=['date'])
transactions=transactions.dropna(subset=['transaction_date'])

products.duplicated().sum()
products=products.drop_duplicates()

transactions.duplicated().sum()
transactions=transactions.drop_duplicates()

customers.duplicated().sum()
customers=customers.drop_duplicates()

products.isna().sum()

products["price_unit"] = products.groupby("sku")["price_unit"].transform(
    lambda x: x.fillna(x.mean()))
products["delivery_days"] = products["delivery_days"].fillna(products["delivery_days"].mean())

transactions.isna().sum()
transactions["discount_pct"] = transactions["discount_pct"].fillna(0)
transactions["unit_price"] = transactions["unit_price"].fillna(
    transactions["final_price"] / transactions["quantity"])


customers.isna().sum()
customers["region"] = customers["region"].fillna("Unknown")
customers["channel_preference"] = customers["channel_preference"].fillna("Unknown")

pcols = ["region","sku","brand","segment","category","channel","pack_type"]

# Use apply to run the string methods on each column in the list
products[pcols] = products[pcols].apply(lambda x: x.str.lower().str.strip())

tcols = ["region","invoice_id","customer_id","sku","channel"]

transactions[tcols] = transactions[tcols].apply(lambda x: x.str.lower().str.strip())

ccols = ["region","customer_id","customer_type","channel_preference"]

customers[ccols] = customers[ccols].apply(lambda x: x.str.lower().str.strip())    

print("FMCG:", products.shape)
print("Transactions:", transactions.shape)
print("Customers:", customers.shape)

products.info()
transactions.info()
customers.info()

products["year_month"] = products["date"].dt.to_period("M").astype(str)

transactions["year_month"] = transactions["transaction_date"].dt.to_period("M").astype(str)

products.to_csv('products_clean.csv',index=False)
transactions.to_csv('transactions_clean.csv',index=False)

products_monthly = (
    products
    .groupby(["sku", "year_month"])
    .agg(
        units_sold=("units_sold", "sum"),
        delivered_qty=("delivered_qty", "sum"),
        stock_available=("stock_available", "mean"),
        price_unit=("price_unit", "mean"),
        promotion_flag=("promotion_flag", "max"),
        delivery_days=("delivery_days", "mean")
    )
    .reset_index()
)

products_monthly.head()
products_monthly.shape

transactions_monthly = (
    transactions
    .groupby(["sku", "year_month"])
    .agg(
        revenue=("final_price", "sum"),
        quantity=("quantity", "sum"),
        unique_customers=("customer_id", "nunique"),
        avg_discount_pct=("discount_pct", "mean"),
        avg_unit_price=("unit_price", "mean")
    )
    .reset_index()
)

transactions_monthly.head()
transactions_monthly.shape

monthly_master = transactions_monthly.merge(
    products_monthly,
    on=["sku", "year_month"],
    how="left"
)

monthly_master.head()
monthly_master.shape

monthly_master.isna().sum()

monthly_master["units_sold"] = monthly_master["units_sold"].fillna(
    monthly_master["quantity"])

monthly_master["delivered_qty"] = monthly_master["delivered_qty"].fillna(0)

monthly_master["stock_available"] = (
    monthly_master
    .groupby("sku")["stock_available"]
    .transform(lambda x: x.fillna(x.mean())))

monthly_master["price_unit"] = monthly_master["price_unit"].fillna(
    monthly_master["avg_unit_price"])

monthly_master["promotion_flag"] = monthly_master["promotion_flag"].fillna(0)

monthly_master["delivery_days"] = (
    monthly_master
    .groupby("sku")["delivery_days"]
    .transform(lambda x: x.fillna(x.mean())))

monthly_master["sku"].nunique()

monthly_master = monthly_master.sort_values(
    by=["sku", "year_month"]
).reset_index(drop=True)

monthly_master.groupby("sku")["year_month"].head()

monthly_master["year_month_dt"] = pd.to_datetime(
    monthly_master["year_month"])

monthly_master["launch_month"] = (
    monthly_master
    .groupby("sku")["year_month_dt"]
    .transform("min"))

monthly_master["product_age_months"] = (
    (monthly_master["year_month_dt"].dt.year - monthly_master["launch_month"].dt.year) * 12 +
    (monthly_master["year_month_dt"].dt.month - monthly_master["launch_month"].dt.month))

monthly_master[["sku", "year_month", "product_age_months"]].head(10)

monthly_master[monthly_master["sku"] == monthly_master["sku"].iloc[0]][
    ["year_month", "product_age_months"]].head(10)

#calculates the percentage change from one time period to the next.
monthly_master["sales_growth_rate"] = (
    monthly_master
    .groupby("sku")["units_sold"]
    .pct_change())

monthly_master["sales_growth_rate"] = monthly_master["sales_growth_rate"].fillna(0)

monthly_master["rolling_avg_sales_3m"] = (
    monthly_master
    .groupby("sku")["units_sold"]
    .transform(lambda x: x.rolling(window=3, min_periods=1).mean()))

monthly_master["rolling_avg_sales_6m"] = (
    monthly_master
    .groupby("sku")["units_sold"]
    .transform(lambda x: x.rolling(window=6, min_periods=1).mean()))

def consecutive_negative_growth(series):
    count = 0
    result = []
    for val in series:
        if val < 0:
            count += 1
        else:
            count = 0
        result.append(count)
    return result

monthly_master["consecutive_negative_growth_count"] = (
    monthly_master
    .groupby("sku")["sales_growth_rate"]
    .transform(consecutive_negative_growth))

monthly_master[["sku", "year_month", "units_sold", "sales_growth_rate"]].head(10)

monthly_master[["units_sold", "rolling_avg_sales_3m", "rolling_avg_sales_6m"]].head(10)

monthly_master[
    monthly_master["consecutive_negative_growth_count"] > 0
][["sku", "year_month", "sales_growth_rate", "consecutive_negative_growth_count"]].head(10)


monthly_master[monthly_master["sku"] == "ju-021"][
    ["year_month", "units_sold", "sales_growth_rate", "consecutive_negative_growth_count"]
]

monthly_master["sales_std_dev_3m"] = (
    monthly_master
    .groupby("sku")["units_sold"]
    .transform(lambda x: x.rolling(window=3, min_periods=1).std()))

monthly_master["sales_std_dev_3m"] = monthly_master["sales_std_dev_3m"].fillna(0)

monthly_master["coefficient_of_variation"] = (
    monthly_master["sales_std_dev_3m"] /
    monthly_master["rolling_avg_sales_3m"])

monthly_master["coefficient_of_variation"] = monthly_master["coefficient_of_variation"].fillna(0)

#Growth volatility measures how unstable or inconsistent the sales growth rate is over time, std_dev stable growth ad if std_dev high then unstable
monthly_master["growth_volatility_3m"] = (
    monthly_master
    .groupby("sku")["sales_growth_rate"]
    .transform(lambda x: x.rolling(window=3, min_periods=1).std()))

monthly_master["growth_volatility_3m"] = monthly_master["growth_volatility_3m"].fillna(0)

monthly_master["peak_sales_value"] = (
    monthly_master
    .groupby("sku")["units_sold"]
    .cummax())

monthly_master["drop_from_peak_percentage"] = (
    (monthly_master["peak_sales_value"] - monthly_master["units_sold"]) /
    monthly_master["peak_sales_value"])

monthly_master["drop_from_peak_percentage"] = monthly_master["drop_from_peak_percentage"].fillna(0)

monthly_master["is_peak_month"] = (
    monthly_master["units_sold"] == monthly_master["peak_sales_value"]
).astype(int)

def time_since_last_peak(is_peak_series):
    counter = 0
    result = []
    for is_peak in is_peak_series:
        if is_peak == 1:
            counter = 0
        else:
            counter += 1
        result.append(counter)
    return result

monthly_master["time_since_peak"] = (
    monthly_master
    .groupby("sku")["is_peak_month"]
    .transform(time_since_last_peak))


monthly_master["sales_momentum"] = (
    monthly_master
    .groupby("sku")["sales_growth_rate"]
    .diff())

monthly_master["sales_momentum"] = monthly_master["sales_momentum"].fillna(0)

monthly_master[["units_sold", "peak_sales_value", "drop_from_peak_percentage"]].head(10)

monthly_master[
    monthly_master["time_since_peak"] > 0
][["sku", "year_month", "time_since_peak"]].head(10)

monthly_master[["sales_growth_rate", "sales_momentum"]].head(10)

monthly_master[
    monthly_master["sku"] == "ju-021"
][["year_month", "units_sold", "is_peak_month", "time_since_peak"]]

sku_id = "ju-021"

sku_df = monthly_master[monthly_master["sku"] == sku_id].copy()

sku_df.shape
sku_df[["year_month", "units_sold"]].head()

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.plot(sku_df["year_month"], sku_df["units_sold"], marker="o", label="Units Sold")
plt.plot(sku_df["year_month"], sku_df["rolling_avg_sales_3m"], label="3M Rolling Avg")
plt.plot(sku_df["year_month"], sku_df["rolling_avg_sales_6m"], label="6M Rolling Avg")

plt.xticks(rotation=45)
plt.title(f"Sales Lifecycle Trend for SKU {sku_id}")
plt.xlabel("Year-Month")
plt.ylabel("Units Sold")
plt.legend()
plt.tight_layout()
plt.show()




plt.figure(figsize=(12, 4))

plt.bar(sku_df["year_month"], sku_df["sales_growth_rate"])
plt.axhline(0)

plt.xticks(rotation=45)
plt.title(f"Monthly Sales Growth Rate for SKU {sku_id}")
plt.xlabel("Year-Month")
plt.ylabel("Growth Rate")
plt.tight_layout()
plt.show()



plt.figure(figsize=(12, 4))

plt.plot(
    sku_df["year_month"],
    sku_df["drop_from_peak_percentage"],
    marker="o",
    label="Drop From Peak (%)"
)

plt.xticks(rotation=45)
plt.title(f"Decline Pressure for SKU {sku_id}")
plt.xlabel("Year-Month")
plt.ylabel("Drop From Peak")
plt.legend()
plt.tight_layout()
plt.show()



plt.figure(figsize=(12, 4))

plt.plot(
    sku_df["year_month"],
    sku_df["time_since_peak"],
    marker="o",
    label="Time Since Peak (Months)"
)

plt.xticks(rotation=45)
plt.title(f"Time Since Peak for SKU {sku_id}")
plt.xlabel("Year-Month")
plt.ylabel("Months")
plt.legend()
plt.tight_layout()
plt.show()


sku_median_sales = (
    monthly_master
    .groupby("sku")["units_sold"]
    .median()
    .to_dict())

def label_plc_for_sku(df):
    stage = "Introduction"
    stages = []
    maturity_counter = 0

    for _, row in df.iterrows():

        # ---------- INTRO → GROWTH ----------
        if stage == "Introduction":
            if row["sales_growth_rate"] > 0.15 and row["rolling_avg_sales_3m"] > row["rolling_avg_sales_6m"]:
                stage = "Growth"

        # ---------- GROWTH → MATURITY ----------
        elif stage == "Growth":
            if abs(row["sales_growth_rate"]) < 0.10 and row["time_since_peak"] >= 3:
                stage = "Maturity"
                maturity_counter = 1

        # ---------- MATURITY ----------
        elif stage == "Maturity":
            maturity_counter += 1

            # ---------- MATURITY → DECLINE (STRICT) ----------
            if (
                maturity_counter >= 6
                and row["time_since_peak"] >= 18
                and row["drop_from_peak_percentage"] >= 0.50
                and row["consecutive_negative_growth_count"] >= 4
            ):
                stage = "Decline"

        # ---------- DECLINE (TERMINAL) ----------
        elif stage == "Decline":
            stage = "Decline"

        stages.append(stage)

    df = df.copy()
    df["plc_stage"] = stages
    return df


monthly_master = (
    monthly_master
    .sort_values(["sku", "year_month"])
    .groupby("sku", group_keys=False)
    .apply(label_plc_for_sku))


monthly_master["plc_stage"].value_counts()

monthly_master[
    monthly_master["sku"] == "ju-021"
][["year_month", "units_sold", "plc_stage"]]


feature_cols = [
    "product_age_months",
    "sales_growth_rate",
    "rolling_avg_sales_3m",
    "rolling_avg_sales_6m",
    "consecutive_negative_growth_count",
    "sales_std_dev_3m",
    "coefficient_of_variation",
    "growth_volatility_3m",
    "peak_sales_value",
    "drop_from_peak_percentage",
    "time_since_peak",
    "sales_momentum"]

y = monthly_master["plc_stage"]
X = monthly_master[feature_cols]

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)

label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(label_mapping)

monthly_master = monthly_master.sort_values("year_month_dt")

split_idx = int(len(monthly_master) * 0.8)

X_train = X.iloc[:split_idx]
X_test  = X.iloc[split_idx:]

y_train = y_encoded[:split_idx]
y_test  = y_encoded[split_idx:]

from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=1000
)

model_lr.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

y_pred = model_lr.predict(X_test)

print(classification_report(y_test, y_pred, target_names=le.classes_))

print(confusion_matrix(y_test, y_pred))

coef_df = pd.DataFrame(
    model_lr.coef_,
    columns=feature_cols,
    index=le.classes_)

coef_df.T.sort_values(by="Decline", ascending=False)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print(classification_report(y_test, y_pred_rf, target_names=le.classes_))
print(confusion_matrix(y_test, y_pred_rf))

rf_importance = pd.Series(
    rf_model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

rf_importance

from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix

metrics_lr = {}

metrics_lr["Accuracy"] = accuracy_score(y_test, y_pred)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="macro"
)

metrics_lr["Macro Precision"] = precision
metrics_lr["Macro Recall"] = recall
metrics_lr["Macro F1"] = f1


metrics_rf = {}

metrics_rf["Accuracy"] = accuracy_score(y_test, y_pred_rf)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred_rf, average="macro"
)

metrics_rf["Macro Precision"] = precision
metrics_rf["Macro Recall"] = recall
metrics_rf["Macro F1"] = f1

comparison_df = pd.DataFrame(
    [metrics_lr, metrics_rf],
    index=["Logistic Regression", "Random Forest"])

comparison_df

def decline_metrics(y_true, y_pred, decline_label):
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[decline_label, decline_label]
    fn = cm[decline_label].sum() - tp
    fp = cm[:, decline_label].sum() - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall

decline_label = label_mapping["Decline"]

decline_lr = decline_metrics(y_test, y_pred, decline_label)
decline_rf = decline_metrics(y_test, y_pred_rf, decline_label)

decline_comparison = pd.DataFrame(
    {
        "Precision": [decline_lr[0], decline_rf[0]],
        "Recall": [decline_lr[1], decline_rf[1]]
    },
    index=["Logistic Regression", "Random Forest"]
)

decline_comparison

cm_lr = confusion_matrix(y_test, y_pred)
cm_rf = confusion_matrix(y_test, y_pred_rf)

print("Logistic Regression Confusion Matrix:\n", cm_lr)
print("\nRandom Forest Confusion Matrix:\n", cm_rf)


comparison_plot_df = comparison_df.reset_index().rename(
    columns={"index": "Model"})


comparison_plot_df.set_index("Model").plot(
    kind="bar",
    figsize=(10, 6))

plt.title("PLC Stage Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.grid(axis="y")
plt.tight_layout()
plt.show()


decline_plot_df = decline_comparison.reset_index().rename(
    columns={"index": "Model"})


decline_plot_df.set_index("Model").plot(
    kind="bar",
    figsize=(8, 5))

plt.title("Decline Detection Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.grid(axis="y")
plt.tight_layout()
plt.show()


monthly_master = monthly_master.sort_values(["sku", "year_month"])

def create_decline_risk_3m(stage_series):
    risk = []
    stages = stage_series.values
    n = len(stages)

    for i in range(n):
        future_window = stages[i+1 : i+4]  # next 3 months
        if "Decline" in future_window:
            risk.append(1)
        else:
            risk.append(0)
    return risk

monthly_master["decline_risk_3m"] = (
    monthly_master
    .groupby("sku")["plc_stage"]
    .transform(create_decline_risk_3m))

monthly_master["decline_risk_3m"].value_counts()

risk_features = [
    "sales_growth_rate",
    "rolling_avg_sales_3m",
    "rolling_avg_sales_6m",
    "consecutive_negative_growth_count",
    "sales_std_dev_3m",
    "growth_volatility_3m",
    "drop_from_peak_percentage",
    "time_since_peak",
    "sales_momentum",
    "product_age_months"]

X_risk = monthly_master[risk_features]
y_risk = monthly_master["decline_risk_3m"]


monthly_master = monthly_master.sort_values("year_month_dt")

split_idx = int(len(monthly_master) * 0.8)

X_train_risk = X_risk.iloc[:split_idx]
X_test_risk  = X_risk.iloc[split_idx:]

y_train_risk = y_risk.iloc[:split_idx]
y_test_risk  = y_risk.iloc[split_idx:]


risk_lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced")

risk_lr.fit(X_train_risk, y_train_risk)


y_pred_risk_lr = risk_lr.predict(X_test_risk)

print(classification_report(y_test_risk, y_pred_risk_lr))
print(confusion_matrix(y_test_risk, y_pred_risk_lr))


risk_rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42)

risk_rf.fit(X_train_risk, y_train_risk)


y_pred_risk_rf = risk_rf.predict(X_test_risk)

print(classification_report(y_test_risk, y_pred_risk_rf))
print(confusion_matrix(y_test_risk, y_pred_risk_rf))


risk_importance = pd.Series(
    risk_rf.feature_importances_,
    index=risk_features
).sort_values(ascending=False)

risk_importance


risk_metrics_lr = {}

risk_metrics_lr["Accuracy"] = accuracy_score(y_test_risk, y_pred_risk_lr)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test_risk, y_pred_risk_lr, average="binary"
)

risk_metrics_lr["Precision"] = precision
risk_metrics_lr["Recall"] = recall
risk_metrics_lr["F1"] = f1


risk_metrics_rf = {}

risk_metrics_rf["Accuracy"] = accuracy_score(y_test_risk, y_pred_risk_rf)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test_risk, y_pred_risk_rf, average="binary"
)

risk_metrics_rf["Precision"] = precision
risk_metrics_rf["Recall"] = recall
risk_metrics_rf["F1"] = f1


risk_comparison_df = pd.DataFrame(
    [risk_metrics_lr, risk_metrics_rf],
    index=["Logistic Regression", "Random Forest"])

risk_comparison_df


risk_comparison_df.plot(
    kind="bar",
    figsize=(10, 6))

plt.title("Decline Risk Model Performance Comparison (3-Month Horizon)")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.tight_layout()
plt.show()


risk_comparison_df[["Precision", "Recall"]].plot(
    kind="bar",
    figsize=(8, 5))

plt.title("Decline Risk Precision vs Recall Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.tight_layout()
plt.show()


risk_importance.sort_values().plot(
    kind="barh",
    figsize=(8, 6))

plt.title("Feature Importance for Decline Risk Prediction (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


monthly_master["decline_risk_prob"] = risk_rf.predict_proba(X_risk)[:, 1]

def assign_risk_level(prob):
    if prob >= 0.70:
        return "High Risk"
    elif prob >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"

monthly_master["decline_risk_level"] = monthly_master["decline_risk_prob"].apply(assign_risk_level)

def decline_reason(row):
    reasons = []

    if row["drop_from_peak_percentage"] >= 0.40:
        reasons.append("Significant drop from peak sales")

    if row["time_since_peak"] >= 15:
        reasons.append("Long time since last sales peak")

    if row["consecutive_negative_growth_count"] >= 3:
        reasons.append("Sustained negative growth")

    if row["sales_momentum"] < 0:
        reasons.append("Negative sales momentum")

    if row["product_age_months"] >= 24:
        reasons.append("Product aging")

    # ML-based fallback explanation (CRITICAL FIX)
    if not reasons and row["decline_risk_level"] == "High Risk":
        reasons.append("Early decline risk detected by ML model based on combined signals")

    return "; ".join(reasons)

monthly_master["decline_reason"] = monthly_master.apply(decline_reason, axis=1)


def recommend_action(row):
    actions = []

    if "drop from peak" in row["decline_reason"]:
        actions.append("Review pricing and promotions")

    if "negative growth" in row["decline_reason"]:
        actions.append("Increase marketing or bundle offers")

    if "time since last sales peak" in row["decline_reason"]:
        actions.append("Consider product repositioning")

    if "Product aging" in row["decline_reason"]:
        actions.append("Evaluate SKU rationalization or replacement")

    if "Early decline risk detected" in row["decline_reason"]:
        actions.append("Monitor closely and prepare corrective actions")

    if not actions and row["decline_risk_level"] == "High Risk":
        actions.append("Monitor closely and prepare corrective actions")

    return "; ".join(actions)


monthly_master["recommended_action"] = monthly_master.apply(recommend_action, axis=1)


final_output = monthly_master[
    monthly_master["decline_risk_level"] == "High Risk"
][[
    "sku",
    "year_month",
    "plc_stage",
    "decline_risk_prob",
    "decline_risk_level",
    "decline_reason",
    "recommended_action"
]]

final_output.head(10)

'''
import joblib

joblib.dump(rf_model, "plc_stage_rf_model.pkl")
joblib.dump(risk_rf, "decline_risk_rf_model.pkl")

joblib.dump(feature_cols, "plc_feature_cols.pkl")
joblib.dump(risk_features, "risk_feature_cols.pkl")
joblib.dump(le, "plc_label_encoder.pkl")

risk_thresholds = {
    "low": 0.40,
    "high": 0.70}
joblib.dump(risk_thresholds, "risk_thresholds.pkl")

'''

regional_monthly = (
    products
    .assign(year_month=products["date"].dt.to_period("M"))
    .groupby(["sku", "region", "year_month"])["units_sold"]
    .sum()
    .reset_index())

regional_monthly["region_sales_share"] = (
    regional_monthly["units_sold"] /
    regional_monthly.groupby(["sku", "year_month"])["units_sold"].transform("sum"))

regional_monthly["year_month"] = regional_monthly["year_month"].astype(str)

regional_monthly = regional_monthly.merge(
    monthly_master[["sku", "year_month", "decline_risk_prob"]],
    on=["sku", "year_month"],
    how="left")

regional_hotspots = regional_monthly[
    (regional_monthly["decline_risk_prob"] >= 0.7) &
    (regional_monthly["region_sales_share"] >= 0.3)]

def regional_recommendation(row):
    return f"Prioritize corrective actions in {row['region']}"

regional_hotspots["regional_action"] = regional_hotspots.apply(
    regional_recommendation, axis=1)


regional_hotspots_top1 = (
    regional_hotspots
    .sort_values(
        ["sku", "year_month", "region_sales_share"],
        ascending=[True, True, False]
    )
    .groupby(["sku", "year_month"], as_index=False)
    .first())

regional_hotspots_top1["regional_action"] = (
    "Prioritize corrective actions in " + regional_hotspots_top1["region"]
)

monthly_master.to_csv('monthly_master.csv', index=False)

regional_hotspots_top1.to_csv('regional_hotspots_top1.csv', index=False)

final_output["year_month"] = final_output["year_month"].astype(str)
regional_hotspots_top1["year_month"] = regional_hotspots_top1["year_month"].astype(str)

final_output = final_output.merge(
    regional_hotspots_top1[
        ["sku", "year_month", "region", "regional_action"]
    ],
    on=["sku", "year_month"],
    how="left"
)

def combine_recommendations(row):
    actions = []

    if pd.notna(row.get("recommended_action")) and row["recommended_action"].strip():
        actions.append(row["recommended_action"])

    if pd.notna(row.get("regional_action")):
        actions.append(row["regional_action"])

    return "; ".join(actions)

final_output["final_recommendation"] = final_output.apply(
    combine_recommendations,
    axis=1
)

final_output[
    ["sku", "year_month", "plc_stage", "decline_risk_level",
     "region", "final_recommendation"]
].head(10)

transactions[["invoice_id", "sku", "quantity"]].head()

import itertools
from collections import Counter

mba_df = transactions[["invoice_id", "sku", "quantity"]].copy()
mba_df.head()

basket = (
    mba_df
    .groupby(["invoice_id", "sku"])["quantity"]
    .sum()
    .unstack(fill_value=0))

basket = basket.applymap(lambda x: 1 if x > 0 else 0)

print("Basket shape:", basket.shape)
basket.head()

from mlxtend.frequent_patterns import apriori, association_rules

frequent_itemsets = apriori(
    basket,
    min_support=0.005,   # 0.5% baskets
    use_colnames=True)


frequent_itemsets.sort_values("support", ascending=False).head(10)

rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1.0)

rules.head()

rules_filtered = rules[
    (rules["support"] >= 0.005) &
    (rules["confidence"] >= 0.10) &
    (rules["lift"] >= 1.02)
].sort_values("lift", ascending=False)

rules_filtered.head(10)

if rules_filtered.empty:
    print("No strong association rules found — switching to co-occurrence analysis.")
else:
    print("Strong association rules found.")


invoice_items = (
    mba_df
    .groupby("invoice_id")["sku"]
    .apply(list))

pair_counter = Counter()

for items in invoice_items:
    unique_items = sorted(set(items))
    for pair in itertools.combinations(unique_items, 2):
        pair_counter[pair] += 1


co_occurrence_df = (
    pd.DataFrame(pair_counter.items(), columns=["sku_pair", "co_occurrence_count"])
    .sort_values("co_occurrence_count", ascending=False)
)


co_occurrence_df.head(10)

co_occurrence_df["sku_1"] = co_occurrence_df["sku_pair"].apply(lambda x: x[0])
co_occurrence_df["sku_2"] = co_occurrence_df["sku_pair"].apply(lambda x: x[1])

co_occurrence_df = co_occurrence_df.drop(columns="sku_pair")


co_occurrence_df.head(10)

co_occurrence_filtered = co_occurrence_df[
    co_occurrence_df["co_occurrence_count"] >= 20   # adjust if needed
]

co_occurrence_filtered.head(10)

high_risk_skus = final_output["sku"].unique()

mba_for_high_risk = co_occurrence_filtered[
    (co_occurrence_filtered["sku_1"].isin(high_risk_skus)) |
    (co_occurrence_filtered["sku_2"].isin(high_risk_skus))
]

mba_for_high_risk.head(10)

def mba_bundle_recommendation(row, mba_df):
    sku = row["sku"]
    matches = mba_df[
        (mba_df["sku_1"] == sku) | (mba_df["sku_2"] == sku)
    ].head(2)

    if matches.empty:
        return ""

    bundles = set()
    for _, r in matches.iterrows():
        bundles.add(r["sku_1"] if r["sku_1"] != sku else r["sku_2"])

    return "Bundle with: " + ", ".join(bundles)

final_output["mba_recommendation"] = final_output.apply(
    lambda row: mba_bundle_recommendation(row, mba_for_high_risk)
    if row["decline_risk_level"] == "High Risk" else "",
    axis=1)

sku_category_map = (
    products[["sku", "category"]]
    .drop_duplicates()
    .set_index("sku")["category"]
    .to_dict())

sku_brand_map = (
    products[["sku", "brand"]]
    .drop_duplicates()
    .set_index("sku")["brand"]
    .to_dict())

sku_pack_map = (
    products[["sku", "pack_type"]]
    .drop_duplicates()
    .set_index("sku")["pack_type"]
    .to_dict())


bundle_category_df = mba_for_high_risk.copy()

bundle_category_df["brand_1"] = bundle_category_df["sku_1"].map(sku_brand_map)
bundle_category_df["brand_2"] = bundle_category_df["sku_2"].map(sku_brand_map)

bundle_category_df["pack_type_1"] = bundle_category_df["sku_1"].map(sku_pack_map)
bundle_category_df["pack_type_2"] = bundle_category_df["sku_2"].map(sku_pack_map)

bundle_category_df["category_1"] = bundle_category_df["sku_1"].map(sku_category_map)
bundle_category_df["category_2"] = bundle_category_df["sku_2"].map(sku_category_map)

bundle_category_df.head()

def classify_bundle_category(row):
    if pd.isna(row["category_1"]) or pd.isna(row["category_2"]):
        return "Unknown Bundle"

    if row["category_1"] == row["category_2"]:
        return "Same-Category Bundle"
    else:
        return "Cross-Category Bundle"

bundle_category_df["bundle_category"] = bundle_category_df.apply(
    classify_bundle_category, axis=1)

bundle_category_df = bundle_category_df[[
    "sku_1", "brand_1", "pack_type_1",
    "sku_2", "brand_2", "pack_type_2",
    "bundle_category",
    "co_occurrence_count"
]].sort_values("co_occurrence_count", ascending=False)

bundle_category_df["bundle_description"] = (
    bundle_category_df["sku_1"] + " (" +
    bundle_category_df["brand_1"] + ", " +
    bundle_category_df["pack_type_1"] + ") + " +
    bundle_category_df["sku_2"] + " (" +
    bundle_category_df["brand_2"] + ", " +
    bundle_category_df["pack_type_2"] + ")"
)


bundle_category_df.head(10)

bundle_category_df["bundle_category"].value_counts()

bundle_category_df.groupby("bundle_category")["co_occurrence_count"].mean()

def clean_action_for_risk(row):
    if row["decline_risk_level"] == "High Risk":
        return row["recommended_action"].replace(
            "Monitor closely and prepare corrective actions",
            "Immediate corrective action required"
        )
    return row["recommended_action"]

final_output["recommended_action"] = final_output.apply(
    clean_action_for_risk,
    axis=1)

final_output["region"] = final_output["region"].fillna("All Regions")
final_output["regional_action"] = final_output["regional_action"].fillna(
    "Prioritize corrective actions across all regions")

def final_recommendation(row):
    actions = []

    # 1. Decline risk dominates
    if row["decline_risk_level"] == "High Risk":
        actions.append("Urgent decline risk detected")

        if row["recommended_action"]:
            actions.append(row["recommended_action"])

        if row["regional_action"]:
            actions.append(row["regional_action"])

        # MBA only supportive
        if pd.notna(row["mba_recommendation"]):
            actions.append(row["mba_recommendation"])

    else:
        # Non-high risk
        if row["recommended_action"]:
            actions.append(row["recommended_action"])

        if pd.notna(row["mba_recommendation"]):
            actions.append(row["mba_recommendation"])

        if row["regional_action"]:
            actions.append(row["regional_action"])

    return "; ".join(dict.fromkeys(actions))


final_output["final_recommendation"] = final_output.apply(
    final_recommendation,
    axis=1)


final_output[[
    "sku",
    "year_month",
    "plc_stage",
    "decline_risk_level",
    "final_recommendation"
]].head(10)

regional_hotspots.to_csv('regional_hotspots.csv',index=False)
final_output.to_csv('final_output.csv',index=False)

bundle_category_df.to_csv('bundle_category.csv',index=False)

import seaborn as sns

df = regional_hotspots_top1.copy()

df["year_month"] = pd.to_datetime(df["year_month"])

region_sku_count = (
    df
    .groupby("region")["sku"]
    .nunique()
    .reset_index(name="sku_count")
)

region_sku_count = region_sku_count.sort_values(
    "sku_count",
    ascending=False)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=region_sku_count,
    x="region",
    y="sku_count",
    palette="Reds"
)

plt.title(
    "Region-wise Count of High Risk SKUs",
    fontsize=14,
    weight="bold"
)

plt.xlabel("Region")
plt.ylabel("Number of SKUs")

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


selected_month = "2024-02"

df_month = monthly_master[
    monthly_master["year_month"] == selected_month]

risk_distribution = (
    df_month
    .groupby("decline_risk_level")["sku"]
    .nunique()
    .reset_index(name="sku_count"))

risk_order = ["Low Risk", "Medium Risk", "High Risk"]

risk_distribution["decline_risk_level"] = pd.Categorical(
    risk_distribution["decline_risk_level"],
    categories=risk_order,
    ordered=True
)

risk_distribution = risk_distribution.sort_values("decline_risk_level")

plt.figure(figsize=(8, 5))

sns.barplot(
    data=risk_distribution,
    x="decline_risk_level",
    y="sku_count",
    palette=["green", "orange", "red"]
)

plt.title(
    f"Decline Risk Distribution ({selected_month})",
    fontsize=14,
    weight="bold"
)

plt.xlabel("Decline Risk Level")
plt.ylabel("Number of SKUs")

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()