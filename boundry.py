import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


feature_num = 2
gaussianNB = GaussianNB()
logisticRegression = LogisticRegression()
knn = KNeighborsClassifier()
decisionTree = DecisionTreeClassifier()

classifiers = [gaussianNB, logisticRegression, knn, decisionTree]


def generate_data(rows_num, w0, w1, w2):
    X = np.random.uniform(-1, 1, size=(rows_num, feature_num))
    y = np.zeros(rows_num)
    for index, x in enumerate(X):
        # y[index] = np.sign(w1*x[0] + w2*x[1] + w0)
        y[index] = np.sign(math.pow(x[0] - w1,2) + math.pow(x[1] - w2,2) - w0)

    return X, y

def scatter(X, y, filename):
    plt.figure(filename)
    points = dict()
    for index, y_class in enumerate(y):
        if y_class in points:
            points[y_class].append((X[index, 0], X[index, 1]))
        else:
            points[y_class] = [(X[index, 0], X[index, 1])]
    for key, val in points.items():
        plt.scatter(*zip(*val))
    plt.savefig(filename)

def main():
    X_train, y_train = generate_data(4000, 0.4, 0.3, -0.5)
    scatter(X_train, y_train, "generated")
    for classifier in classifiers:
        classifier.fit(X_train, y_train)
        y_predicts = classifier.predict(X_train)
        scatter(X_train, y_predicts, type(classifier).__name__)

if __name__ == '__main__':
    main()