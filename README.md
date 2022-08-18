# Sklearn-Model
Building a Logistic Regression Model with Sklearn
import pandas as pd 
df = pd.read_csv("C:/Users/kepohin/OneDrive/titanic.csv")
df.head()
# Prep Data with Pandas

df['male'] = df['Sex'] == 'male'
df.head()

# we need to make all our Features columns(Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare') numerical: using numpy array
x = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
print(x)

# Now let’s take the target (the Survived column) and store it in a variable y.
y = df['Survived'].values
print(y)

# Build a Logistic Regression Model with Sklearn

# We start by importing the Logistic Regression model: all sklearn are built as classes

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

Now we can use our data that we previously prepared to train the model. The fit method is used for building the model. It takes two arguments: X (the features as a 2d numpy array) and y (the target as a 1d numpy array). # 
For simplicity, let’s first assume that we’re building a Logistic Regression model using just the Fare and Age columns. First we define X to be the feature matrix and y the target array.

x = df[['Age', 'Fare']].values
y = df['Survived'].values

model.fit(x, y)

# Make Predictions with the Model

x = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

model = LogisticRegression()
model.fit(x, y)

# Now we can use the predict method to make predictions.
model.predict(x)
df.head()
# The first passenger in the dataset is:[3, True, 22.0, 1, 0, 7.25]This means the passenger is in Pclass 3, are male, are 22 years old, have 1 sibling/spouse aboard, 0 parents/child aboard, and paid $7.25. Let’s see what the model predicts for this passenger. Note that even with one datapoint, the predict method takes a 2-dimensional numpy array and returns a 1-dimensional numpy array.

# 0
print(model.predict([[3, True, 22.0, 1, 0, 7.25]]))

#array([0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,......])# Conclusion:
The result is 0, which means the model predicts that this passenger did not survive.

# 1
print(model.predict([[1, False, 38.0, 1, 0, 71.28]]))

# Conclusion:
The result is 1, which means the model predicts that this passenger did survive.
# Let’s see what the model predicts for the first 5 rows of data and compare it to our target array. We get the first 5 rows of data with X[:5] and the first 5 values of the target with y[:5].

print(model.predict(x[:5]))

# # Score the Model

# Score the Model
# We can get a sense of how good our model is by counting the number of datapoints it predicts correctly. This is called the accuracy score.

# Let’s create an array that has the predicted y values.
y_pred = model.predict(x)

# Now we create an array of boolean values of whether or not our model predicted each passenger correctly.
# y == y_pred

# To get the number of these that are true, we can use the numpy sum method.
print((y == y_pred).sum())
# This means that of the 887 datapoints, the model makes the correct prediction for 714 of them.
#
# To get the percent correct, we divide this by the total number of passengers. We get the total number of passengers using the shape attribute.
y.shape[0]

print((y == y_pred).sum() / y.shape[0])


# # Score the Model: Exercise
# Assume that y=[0, 0, 0, 1, 1] and the result of model.predict(X) is [0, 0, 1, 1, 0]. What is the expected output of the following code?
#
# model.score(X, y)
from sklearn.metrics import accuracy_score
# True class
y = [0, 0, 0, 1, 1] 
# Predicted class
y_hat = [0, 0, 1, 1, 0]
accuracy_score(y, y_hat)


