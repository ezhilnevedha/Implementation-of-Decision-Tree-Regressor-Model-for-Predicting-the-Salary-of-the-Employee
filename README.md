# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.             
2.Upload the dataset and check for any null values using .isnull() function.            
3.Import LabelEncoder and encode the dataset.           
4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.                
5.Predict the values of arrays.             
6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.           
7.Predict the values of array.           
8.Apply to new unknown values.               

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Ezhil nevedha.K
RegisterNumber:  212223230055
*/
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv('Salary.csv')
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
![373105913-5d43147a-0ef1-4f78-980a-04445613c668](https://github.com/user-attachments/assets/3c5dfb83-31de-4a01-83d2-a55970e7da15)

![373106014-35630bb8-1523-46b0-8497-e1809aaaca8b](https://github.com/user-attachments/assets/9d42122b-db8b-4fe9-b869-be831ade8c28)

![373106155-6381af5a-33ba-48f2-b730-2b001d1ff726](https://github.com/user-attachments/assets/49a6d0f3-016c-4487-9532-a5916309ee40)

![373107486-326a2ce5-de47-48ec-8ef4-455613b98686](https://github.com/user-attachments/assets/43597837-fa38-4b55-b594-c948ecea932a)

![373107868-3f97c3ca-d44c-4717-b1d2-f66afc1c5c37](https://github.com/user-attachments/assets/551bfd1c-58d2-41d3-bf30-9d71b2764ad1)

![373108141-f7c8fb8f-f83a-4354-9776-f14ad5e299f7](https://github.com/user-attachments/assets/17a1417a-dcf2-4ed6-a98b-91eb338e8bf5)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
