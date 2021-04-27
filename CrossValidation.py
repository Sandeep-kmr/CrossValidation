#importing all the important lib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
df = pd.read_csv('Social_Network_Ads.csv')



#seperation of dependent and independt variable

x= df[['Age', 'EstimatedSalary']]
y= df['Purchased']


# Train test split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state =5)
knnclassifier = KNeighborsClassifier(n_neighbors=5)
knnclassifier.fit(x_train, y_train)
y_pred = knnclassifier.predict(x_test)
metrics.accuracy_score(y_pred, y_test)

#I have used KNN Classifier for this model
knnclassifier = KNeighborsClassifier(n_neighbors=5)

#Get Accuracy for 10 combination
AllAccuracy=cross_val_score(knnclassifier , x , y, cv=10 , scoring= 'accuracy')
print("All accuracy : ", AllAccuracy)

#Average of all the accuracy
averageAccuracy=cross_val_score(knnclassifier , x , y, cv=10 , scoring= 'accuracy').mean()
print("Average accuracy: ", averageAccuracy)

# Minimum Accuracy
MinAc=cross_val_score(knnclassifier , x , y, cv=10 , scoring= 'accuracy').min()
print("Min accuracy: ", MinAc)


#Maximum Accuracy
MaxAc=cross_val_score(knnclassifier , x , y, cv=10 , scoring= 'accuracy').max()
print("Max accuracy: ", MaxAc)