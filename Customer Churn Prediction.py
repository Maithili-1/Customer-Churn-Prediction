#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import seaborn as sns
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tkinter import *
from tkinter import messagebox

# Load Data
data = pd.read_csv("Churn_Modelling.csv")

# Data Preprocessing
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
data = pd.get_dummies(data, drop_first=True)
X = data.drop('Exited', axis=1)
y = data['Exited']

# Handle Imbalanced Data
X_res, y_res = SMOTE().fit_resample(X, y)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.20, random_state=42)

# Standardize Data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_res = sc.fit_transform(X_res)

# Train Models
models = {
    'LR': LogisticRegression(),
    'SVM': svm.SVC(),
    'KNN': KNeighborsClassifier(),
    'DT': DecisionTreeClassifier(),
    'RF': RandomForestClassifier(),
    'GBC': GradientBoostingClassifier()
}

# Fit and Evaluate Models
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    results.append({'Model': name, 'Accuracy': acc, 'Precision': prec})

# Save Best Model
best_model = max(results, key=lambda x: x['Accuracy'])['Model']
joblib.dump(models[best_model], 'churn_predict_model')

# Visualization
results_df = pd.DataFrame(results)
sns.barplot(x='Model', y='Accuracy', data=results_df)
sns.barplot(x='Model', y='Precision', data=results_df)

# Tkinter GUI for Predictions
def show_entry_fields():
    try:
        features = [int(e1.get()), int(e2.get()), int(e3.get()), float(e4.get()), int(e5.get()), int(e6.get()), int(e7.get()), float(e8.get())]
        geo = int(e9.get())
        gender = int(e10.get())
        if geo == 1:
            features.extend([1, 0])
        elif geo == 2:
            features.extend([0, 1])
        else:
            features.extend([0, 0])
        features.append(gender)
        model = joblib.load('churn_predict_model')
        result = model.predict(sc.transform([features]))
        message = "The customer will stay" if result == 0 else "The customer will not stay."
        Label(root, text=f"Prediction Result: {message}").grid(row=11)
    except Exception as e:
        messagebox.showerror("Error", str(e))

root = Tk()
root.title("Customer Churn Prediction")

labels = ["Credit Score", "Age", "Tenure", "Balance", "Number of Products", "Has Credit Card", "Is Active Member", "Estimated Salary", "Geography (1=Germany, 2=Spain, 3=France)", "Gender (0=Female, 1=Male)"]
entries = [Entry(root) for _ in labels]

for i, label in enumerate(labels):
    Label(root, text=label).grid(row=i)
    entries[i].grid(row=i, column=1)

[e1, e2, e3, e4, e5, e6, e7, e8, e9, e10] = entries

Button(root, text='Predict', command=show_entry_fields).grid(row=len(labels), columnspan=2)
mainloop()






