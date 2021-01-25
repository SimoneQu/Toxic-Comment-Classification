import os
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from preprocessor import Preprocessor

def train_NB(X_train, Y_train):
    NB_clfs = dict()
    for col in label_cols:
        print(col)
        NB_clfs[col] = MultinomialNB().fit(X_train, Y_train[col])
    return NB_clfs

def get_scores(clf_dict, X_test, Y_test):
    scores = dict()
    for col, clf in clf_dict.items():
        scores[col] = clf.score(X_test, Y_test[col])
    return scores

def pred(clf_dict, new_feat):
    pred = dict()
    for col, clf in clf_dict.items():
        clf = clf_dict[col]
        pred[col] = clf.predict_proba(new_feat)[:,1]
    df_pred = pd.DataFrame(data=pred)
    # df_pred.to_csv("submission_NB_v1.csv", index=False)
    return df_pred


if __name__ == "__main__":
    ## data processing
    pp = Preprocessor()
    X_train, X_test, labels_train, labels_test, test_feat = pp.prep_data()
    label_cols = pp.labels_cols
    print(labels_train.mean())
    print(labels_train.sum())
    print(X_train.shape, X_test.shape, labels_train.shape, labels_test.shape)

    ## Naive Bayes
    NB_clfs = train_NB(X_train, labels_train)
    NB_scores = get_scores(NB_clfs, X_test, labels_test)
    print("NB scores:", NB_scores)
    # {'toxic': 0.9485373373774848, 'severe_toxic': 0.990675055774196, 'obscene': 0.9708720828215477, 'threat': 0.9971674228561401, 'insult': 0.9661594765999048, 'identity_hate': 0.9904745193392325}
