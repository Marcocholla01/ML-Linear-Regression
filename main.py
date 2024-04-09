import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from the CSV file
df = pd.read_csv('./real_estate.csv')

# Summary Statistics
summary = df.describe()
print(summary)

# Histograms of numerical features
df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

# Scatter plots of numerical features against the target variable
plt.figure(figsize=(12, 8))
for i, col in enumerate(df.columns[:-1]): # Excluding the target variable
    plt.subplot(3, 3, i+1)
    plt.scatter(df[col], df['Y house price of unit area'], alpha=0.5)
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('Y house price of unit area')
plt.tight_layout()
plt.show()

# Correlation analysis
correlation_matrix = df.corr()
print(correlation_matrix['Y house price of unit area'])

# Splitting features and target variable
X = df.drop(columns=['Y house price of unit area'])
y = df['Y house price of unit area']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print('=============================================')
print("")
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("")
print("============================================")
print("")

# Plotting predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()
print('')

# Plotting distribution of residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='green')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.show()
