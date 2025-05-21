# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Start the Program.
2. Import the necessary packages.
3. Read the given csv file and display the few contents of the data.
4. Assign the features for x and y respectively.
5. Split the x and y sets into train and test sets.
6. Convert the Alphabetical data to numeric using CountVectorizer.
7. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
8. Find the accuracy of the model.
9. End the Program.
```
## Program:
```
Developed by: Janani S
RegisterNumber: 212224230103
```
```
import pandas as pd

data=pd.read_csv("spam.csv",encoding="Windows-1252")

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
## Data.head()
![Screenshot 2025-05-21 132719](https://github.com/user-attachments/assets/8b15b9b4-83a2-43af-9b65-be330c7bbaf5)
## Data.info()
![Screenshot 2025-05-21 132734](https://github.com/user-attachments/assets/6bec8bca-5286-4c5b-a01a-d38024bdd5a9)
## Data.isnull.sum()
![Screenshot 2025-05-21 132743](https://github.com/user-attachments/assets/02eb23b0-e649-4e58-8d79-0aabf167e84f)
## Y_Pred
![Screenshot 2025-05-21 132756](https://github.com/user-attachments/assets/e2462f1f-e840-4468-b9b8-2c91aaf91eea)
## accuracy
![Screenshot 2025-05-21 132804](https://github.com/user-attachments/assets/0b081d35-4bf2-456a-9ee0-52191a648c44)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
