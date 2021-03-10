import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pickle

#loading titanic data
titanic = pd.read_csv('https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv')

#dealing with NaN values in age
def replace_age(titanic):
    titanic['Age'] = np.where((titanic['Sex']=='male') & (titanic['Age'].isnull()), 30, titanic['Age'])
    titanic['Age'] = np.where((titanic['Sex']=='female') & (titanic['Age'].isnull()), 28, titanic['Age'])
    print(titanic['Age'].isnull().sum())
replace_age(titanic)

#normalazing data at Fare column:
def cut_ouliers_fare(titanic):
    titanic['Fare'] = np.where(titanic['Fare']>100, np.random.randint(45, 60, 1), titanic['Fare'])
    print(titanic['Fare'])
cut_ouliers_fare(titanic)

#turning Female = 1 and male = 0
def change_sex(titanic):
    titanic['Sex'] = np.where(titanic['Sex'] == 'male', 0, titanic['Sex'])
    titanic['Sex'] = np.where(titanic['Sex'] == 'female', 1, titanic['Sex'])
change_sex(titanic)

#droping string columns:
titanic = titanic.drop('PassengerId', axis = 1)
titanic = titanic.drop('Name', axis = 1)
titanic = titanic.drop('Ticket', axis=1)
titanic = titanic.drop('Embarked', axis = 1)
titanic = titanic.drop('Cabin', axis=1)

#spliting data for ML model
x_train, x_test, y_train, y_test = train_test_split(titanic.drop('Survived', axis = 1), titanic['Survived'],
                                                            test_size = 0.25,
                                                            random_state=42)

#training the model
model = LogisticRegression()
model.fit(x_train, y_train)

#saving the model in binary file
with open ('titanic_model', 'wb') as file:
    pickle.dump(model, file)

#loading the model in binary file
with open ('titanic_model', 'rb') as file:
    pickle.load(file)

    #LOOKING FOR METRICS

#testing our model
y_pred = model.predict(x_test)


#confusion matrix
matrix = confusion_matrix(y_test, y_pred)
print(f'True Negative: {matrix[0][0]}')
print(f'True Positive: {matrix[1][1]}')
print(f'False Positive: {matrix[0][1]}')
print(f'False Nagative: {matrix[1][0]}')

#more analysis
print(classification_report(y_test, y_pred))


#making the predition of this person would be alive or not
user = pd.DataFrame({'Pclass': 2,	'Sex':1, 'Age':22, 'SibSp':0, 'Parch':1, 'Fare':20.500}, index=[0]) 
user_prob = model.predict_proba(user)
prob = (user_prob[0][1])*100
print(prob)
