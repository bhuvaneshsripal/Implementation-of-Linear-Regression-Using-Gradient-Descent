# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset from a CSV file and separate the features and target variable, encoding any categorical variables as needed.
2. Scale the features using a standard scaler to normalize the data.
3. Initialize model parameters (theta) and add an intercept term to the feature set.
4. Train the linear regression model using gradient descent by iterating through a specified number of iterations to minimize the cost function.
5. Make predictions on new data by transforming it using the same scaling and encoding applied to the training data.

## Program:
#### Program to implement the linear regression using gradient descent.
#### Developed by: BHUVANESH S
#### RegisterNumber: 212225040049
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(x1,y, learning_rate=0.1):
    x = np.c_[np.ones(len(x1)),x1]
    theta=np.zeros((x.shape[1], 1))
    num_iters=1000
    for _ in range(num_iters):
        predictions = x.dot(theta)
        errors = predictions-y
        theta -= learning_rate*(1/len(x1))*x.T.dot(errors)
    return theta

data = pd.read_csv(r"50_Startups.csv")
x = data.iloc[:, :-1].values
x = x[:, [0,1,2]]
y = data.iloc[:, -1].values.reshape(-1,1)

scaler_x = StandardScaler()
x_scaled = scaler_x.fit_transform(x)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)
print(x_scaled)

theta = linear_regression(x_scaled,y_scaled)
print("Theta:")
print(theta)

new_data = np.array([[165349.2, 136897.8, 471784.1]])
new_scaled = scaler_x.transform(new_data)
new_scaled = np.c_[np.ones((1, 1)), new_scaled]
prediction_scaled = new_scaled.dot(theta)
prediction = scaler_y.inverse_transform(prediction_scaled)

print("Scaled Prediction:",prediction_scaled)
print("Predicted Profit:",prediction)
```

## Output:
<img width="532" height="862" alt="image" src="https://github.com/user-attachments/assets/c8ea9c24-ee39-488e-8256-838b25cefd41" />
<img width="452" height="96" alt="image" src="https://github.com/user-attachments/assets/864f291c-d123-4436-b15f-ffd13d500f50" />
<img width="507" height="46" alt="image" src="https://github.com/user-attachments/assets/ebf67ca5-934f-4763-8a82-44cf73b388be" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
