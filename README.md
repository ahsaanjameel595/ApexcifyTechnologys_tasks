# ApexcifyTechnologys_tasks
Completed 3+ tasks demonstrating Python, data preprocessing, EDA, and ML concepts. Includes hands-on projects with clean code, visualizations, and analysis, showcasing practical skills and problem-solving in real-world datasetmonthly sales prediction:
# monthly sales prediction:
# Monthly Sales Prediction

Predict monthly sales trends using Python and Machine Learning. This project includes **data preprocessing, feature engineering, exploratory data analysis (EDA), and Linear Regression modeling** to forecast future sales and extract insights.

---

## **Code**
Implemented in Python using Jupyter Notebook. Key steps:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('data/monthly_sales.csv')

# Feature engineering
df['Year'] = pd.to_datetime(df['Date']).dt.year
df['Month'] = pd.to_datetime(df['Date']).dt.month

# Train/test split
X = df[['Year', 'Month']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Visualization
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Monthly Sales Prediction')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.show()

# sales



