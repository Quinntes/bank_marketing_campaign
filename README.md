### **Predicting Potential Deposit Customers Using Machine Learning to Optimize Telemarketing Campaigns**

---

## **Business Problem**

The marketing team has data regarding telemarketing campaigns for fixed-term deposit products. This data includes customers who subscribed to the deposit and those who did not after receiving a telemarketing offer.

### **Problem:**

The marketing team struggles to predict which customers will subscribe to the deposit, even though they have access to data on their interactions with the bank (e.g., age, job status, account balance, etc.).

### **Metrics:**

#### **False Positives (FP):**

**Definition**: The model predicts that a customer will subscribe to a deposit, but in reality, they do not.

**Consequences**:

* If the model predicts a customer will subscribe to a deposit, the marketing team contacts that customer, leading to unnecessary telemarketing costs.
* **Cost**: Each telemarketing call costs resources, such as labor and other resources. For instance, we assume the cost of telemarketing is Rp 50,000 per call.

#### **False Negatives (FN):**

**Definition**: The model predicts that a customer will not subscribe to a deposit, but in reality, they do.

**Consequences**:

* If the model predicts that a customer will not subscribe, the marketing team does not contact them, resulting in lost opportunities to increase the campaign's conversion rate.
* **Cost**: Loss of potential revenue because customers who could potentially subscribe to a deposit are not contacted. We assume the opportunity cost of not contacting a potential customer is Rp 2,000,000 per customer who could have subscribed.

### **Cost Comparison:**

The cost of telemarketing to a customer who is unlikely to subscribe (FP) is considered lower than the opportunity cost of losing a potential customer who would have subscribed (FN). Therefore, the cost of FN is higher than FP, making detecting customers who will subscribe to a deposit a top priority.

### **The Metric Used:**

**Recall:**

* The primary focus is to detect as many potential deposit customers as possible. Avoiding false negatives is crucial for maximizing the conversion rate and revenue from the telemarketing campaign.
* Although there will be some false positives (customers predicted to subscribe who do not), the telemarketing cost is more manageable and not as high as the lost opportunity from missed potential customers.

---

## **Data Understanding & Preparation**

### **1. Read Dataset**

The dataset is read from a CSV file and stored in a DataFrame. This dataset contains information about the telemarketing campaign for fixed-term deposits, with the goal of predicting whether a customer will subscribe to the deposit or not.

```python
df = pd.read_csv('data_bank_marketing_campaign.csv')
```

### **2. Creating a Backup**

A backup of the dataset is made to ensure the original data remains intact, so that any changes or cleaning done to the data will not affect the original dataset.

```python
df_copy = df.copy()
```

### **3. Data Overview**

An overview of the dataset is presented, showing the structure of the data, column types, and general information.

```python
df_copy.info()
```

### **4. Selecting Relevant Columns for Analysis**

Columns that are relevant to the analysis and modeling process are selected. The target column, which indicates whether a customer subscribed to the deposit or not, is also chosen for prediction.

```python
df_copy = df_copy[['job', 'housing', 'loan', 'contact', 'month', 'poutcome', 'age', 'balance', 'campaign', 'pdays', 'deposit']]
```

---

## **Exploratory Data Analysis (EDA)**

EDA is performed to explore the dataset, examine feature distributions, and identify potential patterns that can help improve the predictive model.

### **Target Variable Analysis**

The distribution of the target variable (`deposit`) is checked to ensure that the dataset is balanced and appropriate for classification.

```python
perc = (df_copy['deposit'].value_counts(normalize=True) * 100).round(2)
```

---

## **Feature Importance Analysis**

After training the model using Random Forest, we analyze the importance of each feature to understand which features have the most significant impact on the model's decision-making.

```python
feature_names = random_search.best_estimator_.named_steps['Preprocessing'].get_feature_names_out()
importances = random_search.best_estimator_.named_steps['model'].feature_importances_

feat_imp = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)
```

---

## **Model Evaluation & Tuning**

### **1. Model Definition**

* **Random Forest** is chosen for its ability to handle non-linear data and feature interactions automatically.
* **XGBoost** and **Gradient Boosting** are used for their effective performance in classification tasks.
* **Logistic Regression** serves as a baseline model for comparison with more complex models.

### **2. Hyperparameter Tuning**

Hyperparameter tuning is conducted to find the optimal parameter combination for the **Random Forest** model using **RandomizedSearchCV**.

```python
random_search.fit(X_train, y_train)
```

### **3. Final Model Evaluation**

After tuning the hyperparameters, the model is evaluated on the test dataset, and performance metrics (such as recall, ROC-AUC, and F2-Score) are computed.

---

## **Conclusion & Recommendations**

### **Conclusion:**

1. **Feature Importance**: Features like `scale__balance` (account balance) and `scale__age` (age) significantly impact predicting potential deposit customers. Therefore, the marketing team should prioritize customers with higher balances and older ages.
2. **Model Performance**: The **Random Forest** model, with a **threshold of 0.2**, provides a **recall of 0.9186**, striking a good balance between detecting deposit customers and managing false positives. The model is effective and cost-efficient for telemarketing.
3. **Threshold Tuning**: By adjusting the threshold to 0.2, the model achieves a high recall (0.9186), ensuring effective detection of potential deposit customers while keeping false positives manageable.
4. **Testing**: While the model performs well on test data, some false negatives remain, indicating room for further improvement, especially in capturing all potential deposit customers.

### **Recommendations:**

1. **Focus on Customers with Higher Balances and Older Ages**:
   Customers with higher account balances and older ages show a stronger likelihood of subscribing to a deposit. Marketing should prioritize these groups for better conversion rates.

2. **Optimize Telemarketing Timing**:
   Certain months such as **March**, **October**, and **September** showed higher conversion rates. The marketing team should optimize the campaign schedule by focusing efforts during these months.

3. **Improve Recall for Deposit Customers**:
   The current threshold of **0.2** already provides an optimal balance between recall and precision, but further threshold adjustments or techniques like **oversampling** may help improve recall and minimize false negatives.

4. **Utilize Financial Features for Targeting**:
   Focus on customers with stable financial situations, such as those with loans or homeownership, as they are more likely to invest in deposit products. Targeting these customers will improve the effectiveness of the telemarketing campaign.

---

## **Future Work:**

Further work could explore additional feature engineering and experiment with different models or ensemble methods to further improve accuracy.
