from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
y = np.array([20, 30, 40, 50, 60])


X = x.reshape(-1, 1)


model = LinearRegression()
model.fit(X, y)


print("Prediction for 6 hours:", model.predict([[6]]))
print("w:", model.coef_)
print("b:", model.intercept_)


plt.scatter(x, y, label="Actual Data")
plt.plot(x, model.predict(X), color="red", label="Best Fit Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.legend()
plt.show()
