# Gaussian Naive Bayes
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
# load the iris datasets
#Import the dataset
import pandas as pd 
dataset = pd.read_csv('restaurants.csv')

#dataset = datasets.load_iris()
# fit a Naive Bayes model to the data
model = GaussianNB()
model.fit(dataset.iloc[:90,0:8],dataset.iloc[:90,-1])
#model.fit(dataset.data, dataset.target)
print(model)
# make predictions
#expected = dataset.target
#predicted = model.predict(dataset.data)

expected = dataset.iloc[:90,-1]
predicted = model.predict(dataset.iloc[:90,0:8])
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
