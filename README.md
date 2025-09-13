<center>
    <h1 style="font-size: 36px; font-weight: bold;">Predicting Potential Deposit Customers Using Machine Learning to Optimize Telemarketing Campaigns</h1>
</center>

## Business Problem

The marketing team has data regarding telemarketing campaigns for fixed-term deposit products. This data includes customers who subscribed to the deposit and those who did not after receiving a telemarketing offer.

### Problem:

The marketing team does not always know which customers will subscribe to the deposit, even though they have data on their interactions with the bank (e.g., age, job status, account balance, etc.).

### Metrics:

#### False Positives (FP):

ML predicts that a customer will subscribe to a deposit, but in reality, they do not.
**Consequences:**

* If the model predicts a customer will subscribe to a deposit, the marketing team contacts that customer, leading to unnecessary telemarketing costs.
* **Cost:** Each telemarketing call costs resources, such as labor and other resources. For instance, we assume the cost of telemarketing is Rp 50,000 per call.

#### False Negatives (FN):

ML predicts a customer will not subscribe to a deposit, but in reality, they do.
**Consequences:**

* If the model predicts that a customer will not subscribe, the marketing team does not contact them. This results in lost opportunities to increase the campaign's conversion rate.
* **Cost:** Loss of potential revenue because customers who could potentially subscribe to a deposit are not contacted. We assume the opportunity cost of not contacting a potential customer is Rp 2,000,000 per customer who could have subscribed.

### Cost Comparison:

The cost of doing telemarketing to a customer who is unlikely to subscribe (FP) is considered lower than the opportunity cost of losing a potential customer who would have subscribed (FN). Therefore, the cost of FN is higher than FP. This makes detecting customers who will subscribe to a deposit a top priority.

### The Metric Used:

**Recall:**

* The primary focus is to detect as many potential deposit customers as possible. Avoiding false negatives is crucial for maximizing the conversion rate and revenue from the telemarketing campaign.
* Although there will be some false positives (customers predicted to subscribe who do not), the telemarketing cost is more manageable and not as high as the lost opportunity from missed potential customers.

---

## Data Understanding & Preparation

In this phase, the dataset is prepared for analysis. This involves understanding the data and cleaning it to ensure it is of good quality before being used for classification modeling.

### 1. Read Dataset

The dataset is read from a CSV file and stored in a DataFrame. This dataset contains information about the telemarketing campaign for fixed-term deposits, with the goal of predicting whether a customer will subscribe to the deposit or not.

```python
df = pd.read_csv('data_bank_marketing_campaign.csv')
```

### 2. Creating a Backup

A backup of the dataset is made to ensure the original data remains intact, so that any changes or cleaning done to the data will not affect the original dataset.

```python
df_copy = df.copy()
```

### 3. Data Overview

An overview of the dataset is presented, showing the structure of the data, column types, and general information.

```python
df_copy.info()
```

### 4. Selecting Relevant Columns for Analysis

Columns that are relevant to the analysis and modeling process are selected. The target column, which indicates whether a customer subscribed to the deposit or not, is also chosen for prediction.

```python
df_copy = df_copy[['job', 'housing', 'loan', 'contact', 'month', 'poutcome', 'age', 'balance', 'campaign', 'pdays', 'deposit']]
```

---

## Exploratory Data Analysis (EDA)

EDA is performed to explore the dataset, examine feature distributions, and identify potential patterns that can help improve the predictive model.

### Target Variable Analysis

The distribution of the target variable (`deposit`) is checked to ensure that the dataset is balanced and appropriate for classification.

```python
perc = (df_copy['deposit'].value_counts(normalize=True) * 100).round(2)
```

---

## Feature Importance Analysis

After training the model using XGBoost and preprocessing features, we analyze the importance of each feature to understand which features have the most significant impact on the model's decision-making.

```python
feature_names = random_search.best_estimator_.named_steps['Preprocessing'].get_feature_names_out()
importances = random_search.best_estimator_.named_steps['model'].feature_importances_

feat_imp = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)
```

---

## Model Evaluation & Tuning

### 1. Model Definition

* **RandomForest** and **XGBoost** are chosen for their strength in handling non-linear data and interactions between variables automatically.
* **Gradient Boosting** is used for its effective performance in classification tasks.
* **Logistic Regression** is used as a baseline model for comparison with more complex models.

### 2. Hyperparameter Tuning

Hyperparameter tuning is conducted to find the optimal parameter combination for the **XGBoost** model using **RandomizedSearchCV**.

```python
random_search.fit(X_train, y_train)
```

### 3. Final Model Evaluation

After tuning the hyperparameters, the model is evaluated on the test dataset, and performance metrics (such as recall, ROC-AUC, and F2-Score) are computed.

---

## Conclusion & Recommendations

### Conclusion:

1. **Feature Importance**: Features like `poutcome_success` play a significant role in predicting potential deposit customers, so marketing should prioritize customers who were involved in previous successful campaigns.
2. **Model Performance**: The **XGBoost** model achieved the highest recall (0.6349), but still struggled with detecting customers likely to deposit. Further improvements in recall are necessary.
3. **Threshold Tuning**: By adjusting the threshold to 0.2, recall improved to 0.91, offering a good balance between detecting deposit customers and managing false positives.
4. **Testing**: The model performed well on test data, but some false negatives remain, indicating the need for further improvements.

### Recommendations:

1. **Focus on Customers with Previous Campaign Success**: Prioritize customers who participated in past successful campaigns, as they are more likely to subscribe.
2. **Optimize Telemarketing Timing**: Campaigns should be optimized for months like March, October, and September, which are more likely to yield higher conversion rates.
3. **Improve Recall for Deposit Customers**: Consider further threshold adjustments or use techniques like oversampling to improve the detection of potential deposit customers.
4. **Utilize Financial Features for Targeting**: Focus more on customers with stable financial situations, such as those with loans or homeownership, as they are more likely to invest in a deposit product.