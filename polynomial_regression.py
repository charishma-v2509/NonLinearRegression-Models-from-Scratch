import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

print("Polynomial regression without using libraries")
def poly(x, y, degree):
    def matrix(x, degree):
        return np.vander(x, degree + 1, increasing=True)

    def polynomial_regression(x, y, degree):
        X = matrix(x, degree)
        coefficients = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
        return coefficients

    def predict(coefficients, x):
        return np.polyval(coefficients[::-1], x)  # Coefficients reversed for polyval

    def metrics(y_true, y_pred):
        n = len(y_true)
        mse = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) / n
        mae = sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n
        rmse = mse ** 0.5
        y_mean = sum(y_true) / n
        ss_total = sum((y_true[i] - y_mean) ** 2 for i in range(n))
        ss_residual = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n))
        r_squared = 1 - (ss_residual / ss_total)
        return mse, mae, rmse, r_squared

    coefficients = polynomial_regression(x, y, degree)
    predictions = predict(coefficients, x)

    mse, mae, rmse, r_squared = metrics(y, predictions)

    print("Estimated Coefficients:", coefficients)
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R² Score:", r_squared)

    plt.scatter(x, y, color='blue', label='Original Data', alpha=0.6)
    plt.plot(np.sort(x), predict(coefficients, np.sort(x)), color='red', label='Custom Polynomial Fit', linewidth=2)
    
    print("Polynomial regression using libraries")

    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(x.reshape(-1, 1))
    model = LinearRegression().fit(X_poly, y)
    y_pred = model.predict(X_poly)
    coefficients_custom = poly(x, y, degree)


    plt.plot(np.sort(x), y_pred[np.argsort(x)], color='green',label='scikit-learn Polynomial Fit', linewidth=2, linestyle='dashed')
    plt.title(f'Comparison of Custom and Library Polynomial Fit (Degree = {degree})')
    plt.xlabel('Sepal Length')
    plt.ylabel('Petal Length')
    plt.legend()
    plt.show()
#test case 1
x = [-2, -1, 0, 1, 2]
y = [-8, -1, 0, 1, 8]
poly(x, y, degree=3)

#test case 2
x = [0, 1, 2, 3, 4]
y = [2, 4, 6, 8, 10]
poly(x, y, degree=1)

#test case 3
x = [0, 1, 2, 3, 4]
y = [1, 4, 9, 16, 25]
poly(x, y, degree=2)

#test case 4
x = [-2, -1, 0, 1, 2]
y = [17, 2, 1, 2, 17]
poly(x, y, degree=4)
