"""
Import the DecisionTreeClassifier model.
"""
#Import the DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

#Import the dataset
import pandas as pd 
dataset = pd.read_csv('restaurants.csv')

"""
Split the data into a training and a testing set
"""
train_features = dataset.iloc[:80,:-1]
"first 80 rows and last comlumn"
test_features = dataset.iloc[80:,:-1]
"after 80th rows and last comlumn"
train_targets = dataset.iloc[:80,-1]
test_targets = dataset.iloc[80:,-1]

"""
Train the model
"""
tree = DecisionTreeClassifier(criterion = 'entropy').fit(train_features,train_targets)

"""
Predict the classes of new, unseen data
"""
prediction = tree.predict(test_features)

"""
Check the accuracy
"""
print("The prediction accuracy is: ",tree.score(test_features,test_targets)*100,"%")
