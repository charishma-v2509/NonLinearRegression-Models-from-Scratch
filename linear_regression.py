import numpy as np
from sklearn.linear_model import LinearRegression

# Training data (hours studied vs scores)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([10, 20, 30, 40, 50])

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict for 6 hours of study
prediction = model.predict([[6]])
print(f"Predicted score for 6 hours of study: {prediction[0]}")
