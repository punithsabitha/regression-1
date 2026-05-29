import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# importing a library for errors of all outputs are averaged with uniform weight
from sklearn.metrics import mean_squared_error, r2_score

# load the diabetes dataset
diabetes_x, diabetes_y = datasets.load_diabetes(return_X_y=True)

# load the diabetes dataset
diabetes_x= diabetes_x[:, np.newaxis, 2]


# Use only one feature
diabetes_x_train = diabetes_x[:-20]
diabetes_x_test = diabetes_x[-20:]
# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_x_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_x_test)

# The coefficients
print('Coefficients: \n', regr.coef_)

# The mean squared error
print('Mean squared error: %.2f'   % mean_squared_error(diabetes_y_test, diabetes_y_pred))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'   % r2_score(diabetes_y_test, diabetes_y_pred))

# plot outputs
plt.scatter(diabetes_x_test, diabetes_y_test, color='black')
plt.plot(diabetes_x_test, diabetes_y_pred, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()

