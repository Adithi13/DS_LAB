import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
dataset = pd.read_csv('advertising.csv')

# Basic info and checks
print(dataset.head(10))
print(dataset.shape)
print(dataset.isna().sum())
print(dataset.duplicated().any())

# Boxplots
fig, axs = plt.subplots(3, figsize=(5, 15))
sns.boxplot(dataset['TV'], ax=axs[0])
sns.boxplot(dataset['Newspaper'], ax=axs[1])
sns.boxplot(dataset['Radio'], ax=axs[2])
plt.tight_layout()
plt.show()

# Distribution and pairplot
sns.displot(dataset['Sales'])
sns.pairplot(dataset, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()

# Heatmap
sns.heatmap(dataset.corr(), annot=True)
plt.show()

# Linear Regression Model
X = dataset[['TV']]
y = dataset['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
model = LinearRegression()
model.fit(X_train, y_train)

# Model summary
print('Intercept:', model.intercept_)
print('Coefficient:', model.coef_[0])
print(f'Regression Equation: Sales = {model.intercept_:.3f} + {model.coef_[0]:.3f} * TV')

# Plot regression line
plt.scatter(X_train, y_train)
plt.plot(X_train, model.intercept_ + model.coef_[0] * X_train, 'r')
plt.show()

# Predictions
y_pred = model.predict(X_test)
print("Prediction for test set:", y_pred)

# Compare actual vs predicted
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)

# Predict a single value
single_pred = model.predict([[56]])
print("Prediction for TV = 56:", single_pred[0])

# Model evaluation
r2 = r2_score(y, model.predict(X))
print(f'R squared value of the model: {r2:.2f}')
