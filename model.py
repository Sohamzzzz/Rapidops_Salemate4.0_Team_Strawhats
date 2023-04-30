import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
print("Enter the Company's sales and advertisment dataset of this particular city")
# Load the dataset from a CSV file
df = pd.read_csv("C:/Users/patel/Downloads/Advertising Budget and Sales.csv")

# Split the dataset into feature variables (X) and target variable (y)
X = df.drop(['Sales '],axis=1)
y = df['Sales ']

# Split the data into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

# Create a linear regression model and fit it to the training data
model = LinearRegression()
#model.fit(x_train, y_train)
model.fit(X,y)
# Print the R-squared score of the model on the testing data
print(model.score(x_test,y_test))

# Find the mean value of the target variable (sales)
mean = df['Sales '].mean()

# Take input from the user for the TV, news, and radio advertisement costs
tvad = int(input("Enter the TV advertisement cost in 1000$: "))
newsad = int(input("Enter the news advertisement cost in 1000$: "))
radioad = int(input("Enter the Radio advertisement cost in 1000$: "))

# Calculate the total marketing expenditure
sum = tvad+newsad+radioad

# Use the model to predict the sales based on the user input and print the result
y_predicted = model.predict([[tvad,radioad,newsad,sum]])
print("The average sales must be:",mean)

# Compare the predicted sales to the mean sales and provide a recommendation based on the difference
if y_predicted <= mean:
    print("In this city the sales is below the average value, So you need to increase the Marketing expenditure in this city")
elif (y_predicted - mean) >5:
    print("In this city the sales is outstanding, So you can cutoff in the marketing expenditure")
else:
    print("In this city the sales is Decent, So the marketing expenditure is sufficient")
