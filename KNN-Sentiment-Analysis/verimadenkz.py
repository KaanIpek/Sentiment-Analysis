from pygments.lexers import go
from sklearn import neighbors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib

from mlxtend.plotting import plot_decision_regions
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import pandas as pd

from sklearn.model_selection import train_test_split

# train Data
trainData =pd.read_excel('grup1_kz_veriseti.xlsx')


# test Data
text=[]
text=(trainData['text'].tolist())


X = text




model = KNeighborsClassifier(n_neighbors=3, p=2)
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, trainData['label'], test_size=test_size, random_state=42)

# vectorize data
vect = CountVectorizer()
X_train = vect.fit_transform(X_train)
X_test = vect.transform(X_test)

# transform data to tfidf form
tfidf = TfidfTransformer()
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)
X_train = X_train.toarray()
X_test = X_test.toarray()




# fit


model.fit(X_train, y_train )

import seaborn as sns
# evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('\n*** ', 'KNN', " test_size ", str(test_size), " acc ", acc, " f1 ", f1, ' ***')
