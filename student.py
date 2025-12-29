import numpy as np
import matplotlib.pyplot as plt
#hoursstudied
x=np.array([1,2,3,4,5])
#marksobtained
y=np.array([20,30,40,50,60])

w=0.0#weight
b=0.0#bias

def predict(x,w,b):
    return w*x+b

def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)

lr = 0.01       
epochs = 1000  

n = len(x)

for i in range(epochs):
    y_pred = predict(x, w, b)

    
    dw = (-2/n) * np.sum(x * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)

  
    w = w - lr * dw
    b = b - lr * db

    if i % 100 == 0:
        print(f"Epoch {i}, Loss: {mse(y, y_pred):.2f}")


print("Final weight (w):", w)
print("Final bias (b):", b)

try:
    hours = float(input("Enter hours studied: "))
except ValueError:
    print("Invalid input. Please enter a numeric value for hours.")
    raise SystemExit(1)

predicted_marks = w * hours + b
print("Predicted marks:", predicted_marks)

plt.scatter(x, y, label="Actual Data")
plt.plot(x, predict(x, w, b), color="red", label="Best Fit Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.legend()
plt.show()



