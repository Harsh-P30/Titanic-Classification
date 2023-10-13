import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


titanic_data = pd.read_csv("titanic.csv")


titanic_data = titanic_data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 
titanic_data = titanic_data.dropna() 


titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'Embarked'], drop_first=True)

X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Example:
new_passenger = pd.DataFrame({
    'Pclass': [3],
    'Age': [30],
    'SibSp': [1],
    'Parch': [0],
    'Fare': [7.25],
    'Sex_male': [1],
    'Embarked_Q': [0],
    'Embarked_S': [1]
})
prediction = clf.predict(new_passenger)
if prediction == 1:
    print("The person is likely to survive.")
else:
    print("The person is not likely to survive.")
