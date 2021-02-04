from sklearn.naive_bayes import MultinomialNB
import pandas as pd


class NaiveBayes(object):
    def __init__(self, classes):
        self.models = dict()
        self.classes = classes
        for cls in self.classes:
            self.models[cls] = MultinomialNB()

    def fit(self, X_train, Y_train):
        for cls in self.classes:
            print(f'fitting {cls} using NB')
            self.models[cls].fit(X_train, Y_train[cls])

    def predict(self, X_test):
        pred = dict()
        for col, clf in self.models.items():
            pred[col] = clf.predict(X_test)
        df_pred = pd.DataFrame(data=pred)
        return df_pred

    def predict_prob(self, X_test):
        pred = dict()
        for col, clf in self.models.items():
            pred[col] = clf.predict_proba(X_test)[:, 1]
        df_pred = pd.DataFrame(data=pred)
        return df_pred

    def score(self, X, Y):
        scores = dict()
        for col, clf in self.models.items():
            scores[col] = clf.score(X, Y[col])
        return scores


