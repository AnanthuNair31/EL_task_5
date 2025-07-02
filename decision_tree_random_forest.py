import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


df = pd.read_csv("heart.csv")
print(df.head())
print(df.describe())
print(df.isnull().sum())


x = df.drop(columns=['target'], axis=1)
y = df['target']

x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=42)

dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(x_train, y_train)

dot = export_graphviz(dtc, out_file= None, feature_names= x.columns, class_names = ["No Disease", "Disease" ] , filled=True, rounded=True, special_characters=True)

graph = graphviz.Source(dot)
graph.render("decision_tree")
graph.view()


train_accuracy  = accuracy_score(y_train, dtc.predict(x_train))
test_accuracy = accuracy_score(y_test, dtc.predict(x_test))

dtc_depth3 =  DecisionTreeClassifier(max_depth = 3,random_state=42)
dtc_depth3.fit(x_train, y_train)
train_accuracy_d3 = accuracy_score(y_train, dtc_depth3.predict(x_train))
test_accuracy_d3 = accuracy_score(y_test, dtc_depth3.predict(x_test))

dtc_depth5 = DecisionTreeClassifier(max_depth = 5,random_state=42)
dtc_depth5.fit(x_train, y_train)
train_accuracy_d5 = accuracy_score(y_train, dtc_depth5.predict(x_train))
test_accuracy_d5 = accuracy_score(y_test, dtc_depth5.predict(x_test))

print({
    "Unrestricted Tree" : {"Train Accuracy" : train_accuracy, "Test Accuracy" : test_accuracy},
    "Maximum Depth : 3" :  {"Train Accuracy" : train_accuracy_d3, "Test Accuracy" : test_accuracy_d3},
    "Maximum Depth : 5" : {"Train Accuracy" : train_accuracy_d5, "Test Accuracy" : test_accuracy_d5}
})

rfc = RandomForestClassifier(n_estimators = 100 , random_state=42)
rfc.fit(x_train, y_train)

rfc_train_ac = accuracy_score(y_train, rfc.predict(x_train))
rfc_test_ac = accuracy_score(y_test, rfc.predict(x_test))

comparison = { "Unrestricted Tree" : {"Train Accuracy" : train_accuracy, "Test Accuracy" : test_accuracy},
               "Decision Tree (max_depth=5)" : {"Train Accuracy" :train_accuracy_d5, "Test Accuracy" : test_accuracy_d5},
               "Random Forest": {"Train Accuracy" : rfc_train_ac, "Test Accuracy" : rfc_test_ac}

}
print(comparison)


importance = rfc.feature_importances_
feature_names = x.columns

indices = np.argsort(importance)[::-1]
sorted_importance = importance[indices]
sorted_feature_names = feature_names[indices]

plt.figure(figsize = (12,10))
plt.barh( sorted_feature_names , sorted_importance , color = "blue")
plt.xlabel("Feature importance")
plt.title("Random Forest Feature importance")
plt.gca().invert_yaxis()
plt.show()


cv_scores = cross_val_score(rfc, x, y, cv=5, scoring='accuracy')

print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())


