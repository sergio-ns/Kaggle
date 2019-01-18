
############### STEP 1

# Import the Pandas library
import pandas as pd
import numpy as np
from sklearn import tree

# Load the train and test datasets to create two DataFrames
train_url = "train.csv"
train = pd.read_csv(train_url)

test_url = "test.csv"
test = pd.read_csv(test_url)

############### STEP 2

# Create the column Child and assign to 'NaN'
train["Child"] = float('NaN')

# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.
train.loc[train["Age"] < 18, "Child"] = 1
train.loc[train["Age"] >= 18, "Child"] = 0


############### STEP 3
# Create a copy of test: test_one
test_one = test.copy()

# Initialize a Survived column to 0
test_one["Survived"] = 0

# Set Survived to 1 if Sex equals "female" and print the `Survived` column from `test_one`
test_one.loc[test_one.Sex=='female', test_one.Survived] = 1


############### STEP 4

train["Age"] = train["Age"].fillna(train["Age"].median())

# Convert the male and female groups to integer form
train.loc[train.Sex=='male',"Sex"]=0
train.loc[train.Sex=='female',"Sex"]=1


# Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")

# Convert the Embarked classes to integer form
train.loc[train.Embarked=='S', "Embarked"] = 0
train.loc[train.Embarked=='C', "Embarked"] = 1
train.loc[train.Embarked=='Q', "Embarked"] = 2


############### STEP 5

# Create the target and features numpy arrays: target, features_one
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values


# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# Look at the importance and score of the included features
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one, target))
