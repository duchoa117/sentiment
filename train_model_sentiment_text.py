# Importing the libraries
from nltk import text
import numpy as np
import pandas as pd
import re

# preprocess text
from preprocess import text_preprocess
from pyvi import ViTokenizer

# vectorize word
from sklearn.feature_extraction.text import CountVectorizer

# split data
from sklearn.model_selection import train_test_split

# model
from sklearn.naive_bayes import (
    BernoulliNB,
    ComplementNB,
    MultinomialNB,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import xgboost


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

# Accuracy, Precision and Recall
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# save model
import pickle

# read raw data
review_path = r'C:\Users\Admin\Desktop\project\sentiment\data\review_restaurant.csv'
dataset = pd.read_csv(review_path)
n, m = dataset.shape

corpus = []
for i in range(0, n):
    review = dataset['review'][i]
    # review = ViTokenizer.tokenize(text_preprocess(review))
    corpus.append(review)
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
# with open("vectorize_word.pkl", "wb") as f:
#     pickle.dump(cv, f)

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

classifiers = {
    "BernoulliNB": BernoulliNB(alpha=0.1),
    "ComplementNB": ComplementNB(),
    "MultinomialNB": MultinomialNB(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(class_weight={0: 1, 1: 2, 2: 1}),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=200, class_weight={0: 1, 1: 2, 2: 1}),
    "LogisticRegression": LogisticRegression(class_weight={0: 1, 1: 2, 2: 1}),
    "MLPClassifier": MLPClassifier(max_iter=1000),
    "AdaBoostClassifier": AdaBoostClassifier(),
    "xgboost": xgboost.XGBClassifier()
}

for classifier_name, classifier in classifiers.items():
    # Predicting the Test set results
    print("classifier name:", classifier_name)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    score1 = accuracy_score(y_test, y_pred)
    score2 = precision_score(y_test, y_pred, average='micro')
    score3 = recall_score(y_test, y_pred, average='micro')
    print("\n")
    print("Accuracy is ", round(score1 * 100, 2), "%")
    print("Precision is ", round(score2, 2))
    print("Recall is ", round(score3, 2))
    print("___________________________________________")
    #   with open(f"save_model/{classifier_name}.pkl", "wb") as f:
    #       pickle.dump(classifier, f)
    result = pd.DataFrame({"review": corpus, "label": y, "y_predict": classifier.predict(X)})
    result.iloc[(result["label"] != result["y_predict"]).sort_values(ascending=False).index, :].to_csv(
        f"save_model/result_{classifier_name}.csv", index=False)
