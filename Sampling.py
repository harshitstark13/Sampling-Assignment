# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.preprocessing import normalize

# Loading the credit card dataset
url = "https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv"
df = pd.read_csv(url)

# Normalizing the 'Amount' column
Amount = normalize([df['Amount']])[0]
df['Amount'] = Amount
df = df.iloc[:, 1:]

# Displaying class distribution in the target variable
print(df.Class.value_counts())

# Applying Random Over-sampling to balance the class distribution
sampler = RandomOverSampler(sampling_strategy=0.95)
x_resample, y_resample = sampler.fit_resample(df.drop('Class', axis=1), df['Class'])

# Displaying the class distribution after over-sampling
print(y_resample.value_counts())

# Creating a DataFrame with the over-sampled data
resample = pd.concat([x_resample, y_resample], axis=1)

# Performing Simple Random Sampling
n = int((1.96*1.96 * 0.5*0.5)/(0.05**2))
SimpleSampling = resample.sample(n=n, random_state=42)

# Splitting the data for model training and testing
X_train, X_test, y_train, y_test = train_test_split(SimpleSampling.drop('Class', axis=1),
                                                    SimpleSampling['Class'],
                                                    test_size=0.2, random_state=42)

# Initializing machine learning models
rf_model = RandomForestClassifier(random_state=42)
lr_model = LogisticRegression()
nb_model = GaussianNB()
dt_model = DecisionTreeClassifier(random_state=42)
knn_model = KNeighborsClassifier()

# Training and evaluating each model
models = [rf_model, lr_model, nb_model, dt_model, knn_model]
model_names = ['Random Forest', 'Logistic Regression', 'Naive Bayes', 'Decision Trees', 'KNN']

for model, name in zip(models, model_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
