import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


data = pd.read_csv('diabetes.csv')


print(data.head())


x = data.iloc[:, 7].values.reshape(-1, 1)
y = data.iloc[:, 6].values

print("X:")
print(x[:5])

print("\nY:")
print(y[:5])


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42
)


classifier = LinearRegression()


classifier.fit(x_train, y_train)

print("\nModel:")
print(classifier)


y_pred = classifier.predict(x_test)

print("\nPredicted values:")
print(y_pred)


print("\nR-squared:", r2_score(y_test, y_pred))
print("Mean squared error:", mean_squared_error(y_test, y_pred))
print("Coefficient:", classifier.coef_)
print("Intercept:", classifier.intercept_)

plt.scatter(
    x_test,
    y_test,
    color='blue',
    label='Actual'
)

# Plot regression line
plt.plot(
    x_test,
    y_pred,
    color='black',
    label='Predicted'
)


plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression Results')
plt.legend()


plt.show()
